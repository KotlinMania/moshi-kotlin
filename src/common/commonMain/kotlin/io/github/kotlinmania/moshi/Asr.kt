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
import io.github.kotlinmania.serde.SerdeError
import io.github.kotlinmania.serde.SerdeException
import io.github.kotlinmania.serde.SerdeResult
import io.github.kotlinmania.serde.serdeCatching
import io.github.kotlinmania.serdecore.de.Deserialize
import io.github.kotlinmania.serdecore.de.Deserializer
import io.github.kotlinmania.serdecore.de.EnumAccess
import io.github.kotlinmania.serdecore.de.F32Deserialize
import io.github.kotlinmania.serdecore.de.F64Deserialize
import io.github.kotlinmania.serdecore.de.I32Deserialize
import io.github.kotlinmania.serdecore.de.MapAccess
import io.github.kotlinmania.serdecore.de.StringDeserialize
import io.github.kotlinmania.serdecore.de.U32Deserialize
import io.github.kotlinmania.serdecore.de.Visitor
import io.github.kotlinmania.serdecore.ser.Serialize
import io.github.kotlinmania.serdecore.ser.Serializer

private val ASR_MSG_VARIANTS: List<String> = listOf("Step", "Word", "EndWord")
private val ASR_STEP_FIELDS: List<String> = listOf("step_idx", "prs")
private val ASR_WORD_FIELDS: List<String> = listOf("tokens", "start_time", "batch_idx")
private val ASR_END_WORD_FIELDS: List<String> = listOf("stop_time", "batch_idx")

public sealed class AsrMsg : Serialize {
    public data class Step(
        public val stepIdx: Int,
        public val prs: List<List<Float>>,
    ) : AsrMsg()

    public data class Word(
        public val tokens: List<UInt>,
        public val startTime: Double,
        public val batchIdx: Int,
    ) : AsrMsg()

    public data class EndWord(
        public val stopTime: Double,
        public val batchIdx: Int,
    ) : AsrMsg()

    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
        serdeCatching {
            when (this) {
                is Step -> {
                    val state = serializer.serializeStructVariant("AsrMsg", 0u, "Step", 2).getOrThrow()
                    state.serializeField("step_idx", stepIdx.asSerialize()).getOrThrow()
                    state
                        .serializeField(
                            "prs",
                            prs.asListSerialize { inner -> inner.asListSerialize { it.asSerialize() } },
                        ).getOrThrow()
                    state.end().getOrThrow()
                }

                is Word -> {
                    val state = serializer.serializeStructVariant("AsrMsg", 1u, "Word", 3).getOrThrow()
                    state.serializeField("tokens", tokens.asListSerialize { it.asSerialize() }).getOrThrow()
                    state.serializeField("start_time", startTime.asSerialize()).getOrThrow()
                    state.serializeField("batch_idx", batchIdx.asSerialize()).getOrThrow()
                    state.end().getOrThrow()
                }

                is EndWord -> {
                    val state = serializer.serializeStructVariant("AsrMsg", 2u, "EndWord", 2).getOrThrow()
                    state.serializeField("stop_time", stopTime.asSerialize()).getOrThrow()
                    state.serializeField("batch_idx", batchIdx.asSerialize()).getOrThrow()
                    state.end().getOrThrow()
                }
            }
        }

    public companion object : Deserialize<AsrMsg> {
        private object StepVariantVisitor : Visitor<AsrMsg> {
            override fun expecting(): String = "struct variant AsrMsg::Step"

            override fun <A> visitMap(access: A): SerdeResult<AsrMsg>
                where A : MapAccess =
                serdeCatching {
                    var stepIdx: Int? = null
                    var prs: List<List<Float>>? = null
                    val prsDe = F32Deserialize.asListDeserialize().asListDeserialize()
                    while (true) {
                        val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                        when (key) {
                            "step_idx" -> stepIdx = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                            "prs" -> prs = access.nextValueSeed(SeedOf(prsDe)).getOrThrow()
                            else -> throw SerdeException(SerdeError.unknownField(key, ASR_STEP_FIELDS))
                        }
                    }
                    Step(
                        stepIdx = stepIdx ?: throw SerdeException(SerdeError.missingField("step_idx")),
                        prs = prs ?: throw SerdeException(SerdeError.missingField("prs")),
                    )
                }
        }

        private object WordVariantVisitor : Visitor<AsrMsg> {
            override fun expecting(): String = "struct variant AsrMsg::Word"

            override fun <A> visitMap(access: A): SerdeResult<AsrMsg>
                where A : MapAccess =
                serdeCatching {
                    var tokens: List<UInt>? = null
                    var startTime: Double? = null
                    var batchIdx: Int? = null
                    val u32ListDe = U32Deserialize.asListDeserialize()
                    while (true) {
                        val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                        when (key) {
                            "tokens" -> tokens = access.nextValueSeed(SeedOf(u32ListDe)).getOrThrow()
                            "start_time" -> startTime = access.nextValueSeed(SeedOf(F64Deserialize)).getOrThrow()
                            "batch_idx" -> batchIdx = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                            else -> throw SerdeException(SerdeError.unknownField(key, ASR_WORD_FIELDS))
                        }
                    }
                    Word(
                        tokens = tokens ?: throw SerdeException(SerdeError.missingField("tokens")),
                        startTime = startTime ?: throw SerdeException(SerdeError.missingField("start_time")),
                        batchIdx = batchIdx ?: throw SerdeException(SerdeError.missingField("batch_idx")),
                    )
                }
        }

        private object EndWordVariantVisitor : Visitor<AsrMsg> {
            override fun expecting(): String = "struct variant AsrMsg::EndWord"

            override fun <A> visitMap(access: A): SerdeResult<AsrMsg>
                where A : MapAccess =
                serdeCatching {
                    var stopTime: Double? = null
                    var batchIdx: Int? = null
                    while (true) {
                        val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                        when (key) {
                            "stop_time" -> stopTime = access.nextValueSeed(SeedOf(F64Deserialize)).getOrThrow()
                            "batch_idx" -> batchIdx = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                            else -> throw SerdeException(SerdeError.unknownField(key, ASR_END_WORD_FIELDS))
                        }
                    }
                    EndWord(
                        stopTime = stopTime ?: throw SerdeException(SerdeError.missingField("stop_time")),
                        batchIdx = batchIdx ?: throw SerdeException(SerdeError.missingField("batch_idx")),
                    )
                }
        }

        override fun <D> deserialize(deserializer: D): SerdeResult<AsrMsg>
            where D : Deserializer =
            deserializer.deserializeEnum(
                "AsrMsg",
                ASR_MSG_VARIANTS,
                object : Visitor<AsrMsg> {
                    override fun expecting(): String = "enum AsrMsg"

                    override fun <A> visitEnum(access: A): SerdeResult<AsrMsg>
                        where A : EnumAccess =
                        serdeCatching {
                            val (variant, variantAccess) = access.variantSeed(SeedOf(StringDeserialize)).getOrThrow()
                            when (variant) {
                                "Step" -> variantAccess.structVariant(ASR_STEP_FIELDS, StepVariantVisitor).getOrThrow()
                                "Word" -> variantAccess.structVariant(ASR_WORD_FIELDS, WordVariantVisitor).getOrThrow()
                                "EndWord" -> variantAccess.structVariant(ASR_END_WORD_FIELDS, EndWordVariantVisitor).getOrThrow()
                                else -> throw SerdeException(SerdeError.unknownVariant(variant, ASR_MSG_VARIANTS))
                            }
                        }
                },
            )
    }
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
