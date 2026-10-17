// port-lint: source asr.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candlenn.gumbelSoftmax
import io.github.kotlinmania.candlenn.softmaxLastDim
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
public sealed class AsrMsg {
    @Serializable
    @SerialName("Step")
    public data class Step(
        @SerialName("step_idx")
        public val stepIdx: Int,
        public val prs: List<List<Float>>,
    ) : AsrMsg()

    @Serializable
    @SerialName("Word")
    public data class Word(
        public val tokens: List<UInt>,
        @SerialName("start_time")
        public val startTime: Double,
        @SerialName("batch_idx")
        public val batchIdx: Int,
    ) : AsrMsg()

    @Serializable
    @SerialName("EndWord")
    public data class EndWord(
        @SerialName("stop_time")
        public val stopTime: Double,
        @SerialName("batch_idx")
        public val batchIdx: Int,
    ) : AsrMsg()
}

public class ItemState internal constructor(
    internal var stepIdx: Int,
    internal var textToken: UInt,
    internal var wordTokens: MutableList<UInt>,
    internal var unendedWord: Boolean,
    internal var lastStopTime: Double,
    internal val audioPadToken: UInt,
    internal val nextCodebooks: UIntArray,
) {
    internal fun reset() {
        stepIdx = 0
        textToken = 0u
        wordTokens.clear()
        unendedWord = false
        lastStopTime = 0.0
        nextCodebooks.fill(audioPadToken)
    }

    public fun textToken(): UInt = textToken

    public fun isFirstStep(): Boolean = stepIdx == 0

    public fun nextToken(
        codebookIdx: Int,
        token: UInt,
    ): UInt {
        val v = nextCodebooks[codebookIdx]
        nextCodebooks[codebookIdx] = token
        return if (isFirstStep()) {
            audioPadToken
        } else {
            v
        }
    }

    internal fun copy(): ItemState =
        ItemState(
            stepIdx = stepIdx,
            textToken = textToken,
            wordTokens = wordTokens.toMutableList(),
            unendedWord = unendedWord,
            lastStopTime = lastStopTime,
            audioPadToken = audioPadToken,
            nextCodebooks = nextCodebooks.copyOf(),
        )
}

public fun interface AsrStepCallback {
    public fun onStep(
        batch: List<ItemState>,
        text: Tensor,
        audioTokens: List<Tensor>,
    )
}

public class AsrState(
    private val asrDelayInTokens: Int,
    private var modelStepIdx: Int,
    private val temperature: Double,
    private val lm: LmModel,
    private val audioTokenizer: Mimi,
    private val device: Device,
    private val batch: List<ItemState>,
) {
    public companion object {
        public fun new(
            batchSize: Int,
            asrDelayInTokens: Int,
            temperature: Double,
            audioTokenizer: Mimi,
            lm: LmModel,
        ): AsrState {
            val textToken = lm.textStartToken()
            val dev = lm.device()
            val padToken = lm.audioPadToken()
            val inCodebooks = lm.inAudioCodebooks()
            val batch =
                List(batchSize) {
                    ItemState(
                        stepIdx = 0,
                        textToken = textToken,
                        wordTokens = mutableListOf(),
                        unendedWord = false,
                        lastStopTime = 0.0,
                        audioPadToken = padToken,
                        nextCodebooks = UIntArray(inCodebooks) { padToken },
                    )
                }
            val s =
                AsrState(
                    asrDelayInTokens = asrDelayInTokens,
                    modelStepIdx = 0,
                    temperature = temperature,
                    lm = lm,
                    audioTokenizer = audioTokenizer,
                    device = dev,
                    batch = batch,
                )
            s.reset()
            return s
        }
    }

    public fun modelStepIdx(): Int = modelStepIdx

    public fun device(): Device = device

    public fun batchSize(): Int = batch.size

    public fun asrDelayInTokens(): Int = asrDelayInTokens

    public fun reset() {
        lm.resetState()
        audioTokenizer.resetState()
        for (s in batch) {
            s.reset()
        }
    }

    public fun stepPcm(
        pcm: Tensor,
        conditions: Condition?,
        mask: StreamMask,
        f: AsrStepCallback,
    ): List<AsrMsg> {
        val audioTokens = audioTokenizer.encodeStep(StreamTensor.fromTensor(pcm), mask)
        val rawTokens = audioTokens.asOption() ?: return emptyList()
        return stepTokens(rawTokens, conditions, mask, f)
    }

    private fun textTokens(): Tensor {
        val bSize = batchSize()
        val textStartToken = lm.textStartToken()
        val dev = lm.device()
        val tokens =
            UIntArray(bSize) { idx ->
                val s = batch[idx]
                if (s.isFirstStep()) textStartToken else s.textToken()
            }
        return Tensor.fromSlice(tokens, Shape(bSize, 1), dev)
    }

    public fun stepTokens(
        audioTokens: Tensor,
        conditions: Condition?,
        mask: StreamMask,
        f: AsrStepCallback,
    ): List<AsrMsg> {
        val (bSize, codebooks, steps) = audioTokens.dims3()
        if (bSize != batchSize()) {
            bail("batch size mismatch: $bSize != ${batchSize()}")
        }
        val words = ArrayList<AsrMsg>()
        for (step in 0 until steps) {
            val stepSlice = audioTokens.narrow(2, step, 1)
            val stepTokens2d = stepSlice.reshape(Shape(bSize, codebooks)).toVec2<UInt>()
            val codebookTensors =
                List(codebooks) { codebookIdx ->
                    val col =
                        UIntArray(bSize) { batchIdx ->
                            if (!mask.isActive(batchIdx)) {
                                0u
                            } else {
                                batch[batchIdx].nextToken(codebookIdx, stepTokens2d[batchIdx][codebookIdx])
                            }
                        }
                    Tensor.fromSlice(col, Shape(bSize, 1), device())
                }
            val text = textTokens()
            f.onStep(batch, text, codebookTensors)
            val optAudioTokens = codebookTensors.map { t -> t as Tensor? }
            val forwardOut = lm.forwardCond(text, optAudioTokens, conditions, mask)
            val textLogits = forwardOut.logits
            val transformerOut = forwardOut.hiddenStates
            modelStepIdx += 1
            val extraHeads = lm.extraHeads(transformerOut)
            val prs = ArrayList<List<Float>>(extraHeads.size)
            for (extraHead in extraHeads) {
                val headProbs =
                    softmaxLastDim(extraHead.toDtype(DType.F32))
                        .narrow(1, 0, 1)
                        .squeeze(1)
                        .narrow(1, 0, 1)
                        .squeeze(1)
                        .toVec1F32()
                        .toList()
                prs.add(headProbs)
            }
            if (prs.isNotEmpty()) {
                words.add(AsrMsg.Step(stepIdx = modelStepIdx(), prs = prs))
            }

            val logitsStep0 = textLogits.narrow(1, 0, 1).squeeze(1)
            val sampledTextTensor =
                if (temperature <= 0.0) {
                    logitsStep0.argmax(logitsStep0.rank() - 1)
                } else {
                    gumbelSoftmax(
                        logitsStep0.toDtype(DType.F32),
                        temperature,
                        D.Minus1,
                    )
                }
            val sampledTextTokens = sampledTextTensor.toVec1<UInt>()
            for (batchIdx in 0 until bSize) {
                if (!mask.isActive(batchIdx)) {
                    continue
                }
                val item = batch[batchIdx]
                val tok = sampledTextTokens[batchIdx]
                item.textToken = tok
                item.stepIdx += 1
                if (item.stepIdx >= asrDelayInTokens) {
                    if (tok == 3u || tok == 0u) {
                        if (item.wordTokens.isNotEmpty()) {
                            val tokens = item.wordTokens.toList()
                            item.wordTokens.clear()
                            words.add(
                                AsrMsg.Word(
                                    tokens = tokens,
                                    startTime = item.lastStopTime,
                                    batchIdx = batchIdx,
                                ),
                            )
                            item.unendedWord = true
                        }
                    } else {
                        item.wordTokens.add(item.textToken)
                    }
                    if (item.textToken == 0u) {
                        val stopTime = (item.stepIdx - asrDelayInTokens).toDouble() / 12.5
                        if (item.unendedWord) {
                            item.unendedWord = false
                            words.add(AsrMsg.EndWord(stopTime = stopTime, batchIdx = batchIdx))
                        }
                        item.lastStopTime = stopTime
                    }
                }
            }
        }
        return words
    }

    public fun resetBatchIdx(batchIdx: Int) {
        if (batchIdx >= batchSize()) {
            bail("batch index out of range: $batchIdx >= ${batchSize()}")
        }
        batch[batchIdx].reset()
        lm.resetBatchIdx(batchIdx, batchSize())
        audioTokenizer.resetBatchIdx(batchIdx, batchSize())
    }
}
