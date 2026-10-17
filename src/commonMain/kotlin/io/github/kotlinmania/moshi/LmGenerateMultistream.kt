// port-lint: source lm_generate_multistream.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.i
import io.github.kotlinmania.candletransformers.generation.LogitsProcessor
import io.github.kotlinmania.serde.SerdeError
import io.github.kotlinmania.serde.SerdeException
import io.github.kotlinmania.serde.SerdeResult
import io.github.kotlinmania.serde.serdeCatching
import io.github.kotlinmania.serdecore.de.Deserialize
import io.github.kotlinmania.serdecore.de.Deserializer
import io.github.kotlinmania.serdecore.de.I32Deserialize
import io.github.kotlinmania.serdecore.de.MapAccess
import io.github.kotlinmania.serdecore.de.StringDeserialize
import io.github.kotlinmania.serdecore.de.U32Deserialize
import io.github.kotlinmania.serdecore.de.Visitor
import io.github.kotlinmania.serdecore.ser.Serialize
import io.github.kotlinmania.serdecore.ser.Serializer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

public const val UNGENERATED: UInt = UInt.MAX_VALUE

private val MULTI_STREAM_CONFIG_FIELDS: List<String> =
    listOf(
        "generated_audio_codebooks",
        "input_audio_codebooks",
        "audio_vocab_size",
        "acoustic_delay",
        "text_pad_token",
        "text_eop_token",
        "text_start_token",
    )

public data class RepetitionPenaltyConfig(
    public val contextSize: Int,
    public val penalty: Float,
)

public data class MultiStreamConfig(
    public val generatedAudioCodebooks: Int,
    public val inputAudioCodebooks: Int,
    public val audioVocabSize: Int,
    public val acousticDelay: Int,
    public val textPadToken: UInt,
    public val textEopToken: UInt,
    public val textStartToken: UInt,
) : Serialize {
    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
        serdeCatching {
            val state = serializer.serializeStruct("Config", 7).getOrThrow()
            state.serializeField("generated_audio_codebooks", generatedAudioCodebooks.asSerialize()).getOrThrow()
            state.serializeField("input_audio_codebooks", inputAudioCodebooks.asSerialize()).getOrThrow()
            state.serializeField("audio_vocab_size", audioVocabSize.asSerialize()).getOrThrow()
            state.serializeField("acoustic_delay", acousticDelay.asSerialize()).getOrThrow()
            state.serializeField("text_pad_token", textPadToken.asSerialize()).getOrThrow()
            state.serializeField("text_eop_token", textEopToken.asSerialize()).getOrThrow()
            state.serializeField("text_start_token", textStartToken.asSerialize()).getOrThrow()
            state.end().getOrThrow()
        }

    public companion object : Deserialize<MultiStreamConfig> {
        override fun <D> deserialize(deserializer: D): SerdeResult<MultiStreamConfig>
            where D : Deserializer =
            deserializer.deserializeStruct(
                "Config",
                MULTI_STREAM_CONFIG_FIELDS,
                object : Visitor<MultiStreamConfig> {
                    override fun expecting(): String = "struct lm_generate_multistream::Config"

                    override fun <A> visitMap(access: A): SerdeResult<MultiStreamConfig>
                        where A : MapAccess =
                        serdeCatching {
                            var generatedAudioCodebooks: Int? = null
                            var inputAudioCodebooks: Int? = null
                            var audioVocabSize: Int? = null
                            var acousticDelay: Int? = null
                            var textPadToken: UInt? = null
                            var textEopToken: UInt? = null
                            var textStartToken: UInt? = null
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "generated_audio_codebooks" -> generatedAudioCodebooks = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "input_audio_codebooks" -> inputAudioCodebooks = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "audio_vocab_size" -> audioVocabSize = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "acoustic_delay" -> acousticDelay = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "text_pad_token" -> textPadToken = access.nextValueSeed(SeedOf(U32Deserialize)).getOrThrow()
                                    "text_eop_token" -> textEopToken = access.nextValueSeed(SeedOf(U32Deserialize)).getOrThrow()
                                    "text_start_token" -> textStartToken = access.nextValueSeed(SeedOf(U32Deserialize)).getOrThrow()
                                    else -> throw SerdeException(SerdeError.unknownField(key, MULTI_STREAM_CONFIG_FIELDS))
                                }
                            }
                            MultiStreamConfig(
                                generatedAudioCodebooks =
                                    generatedAudioCodebooks
                                        ?: throw SerdeException(SerdeError.missingField("generated_audio_codebooks")),
                                inputAudioCodebooks =
                                    inputAudioCodebooks
                                        ?: throw SerdeException(SerdeError.missingField("input_audio_codebooks")),
                                audioVocabSize = audioVocabSize ?: throw SerdeException(SerdeError.missingField("audio_vocab_size")),
                                acousticDelay = acousticDelay ?: throw SerdeException(SerdeError.missingField("acoustic_delay")),
                                textPadToken = textPadToken ?: throw SerdeException(SerdeError.missingField("text_pad_token")),
                                textEopToken = textEopToken ?: throw SerdeException(SerdeError.missingField("text_eop_token")),
                                textStartToken = textStartToken ?: throw SerdeException(SerdeError.missingField("text_start_token")),
                            )
                        }
                },
            )

        public fun v01(): MultiStreamConfig =
            MultiStreamConfig(
                generatedAudioCodebooks = 8,
                inputAudioCodebooks = 8,
                audioVocabSize = 2049,
                acousticDelay = 2,
                textEopToken = 0u,
                textPadToken = 3u,
                textStartToken = 32000u,
            )

        public fun v01TwoWays(): MultiStreamConfig =
            MultiStreamConfig(
                generatedAudioCodebooks = 16,
                inputAudioCodebooks = 0,
                audioVocabSize = 2049,
                acousticDelay = 2,
                textEopToken = 0u,
                textPadToken = 3u,
                textStartToken = 32000u,
            )

        public fun v01OneWay(): MultiStreamConfig =
            MultiStreamConfig(
                generatedAudioCodebooks = 8,
                inputAudioCodebooks = 0,
                audioVocabSize = 2049,
                acousticDelay = 2,
                textEopToken = 0u,
                textPadToken = 3u,
                textStartToken = 32000u,
            )
    }

    public fun audioPadToken(): UInt = (audioVocabSize - 1).toUInt()

    public fun totalAudioCodebooks(): Int = generatedAudioCodebooks + inputAudioCodebooks
}

public class MultiStreamState private constructor(
    private val model: LmModel,
    private val audioTokens: List<UIntArray>,
    private val textTokens: UIntArray,
    private val audioLp: LogitsProcessor,
    private val textLp: LogitsProcessor,
    private var stepIdx: Int,
    private val padMult: Float?,
    private val repetitionPenalty: RepetitionPenaltyConfig?,
    private val forcedAudioTokens: ForcedAudioTokens,
    private var userRating: UInt,
    private val cfgAlpha: Double?,
    private val config: MultiStreamConfig,
) {
    public companion object {
        public fun new(
            model: LmModel,
            maxStepIdx: Int,
            audioLp: LogitsProcessor,
            textLp: LogitsProcessor,
            padMult: Float?,
            repetitionPenalty: RepetitionPenaltyConfig?,
            cfgAlpha: Double?,
            config: MultiStreamConfig,
        ): MultiStreamState {
            val totalLen = maxStepIdx + config.acousticDelay
            val totalCodebooks = config.totalAudioCodebooks()
            val audioTokens =
                List(totalLen) {
                    UIntArray(totalCodebooks) { UNGENERATED }
                }
            val textTokens = UIntArray(totalLen) { UNGENERATED }
            val forcedAudioTokens =
                ForcedAudioTokens.new(
                    acousticDelay = config.acousticDelay,
                    audioPadToken = config.audioPadToken(),
                    streamCodebooks = intArrayOf(8, 8),
                )
            return MultiStreamState(
                model = model,
                audioTokens = audioTokens,
                textTokens = textTokens,
                audioLp = audioLp,
                textLp = textLp,
                stepIdx = 0,
                padMult = padMult,
                repetitionPenalty = repetitionPenalty,
                forcedAudioTokens = forcedAudioTokens,
                userRating = 0u,
                cfgAlpha = cfgAlpha,
                config = config,
            )
        }
    }

    public fun stepIdx(): Int = stepIdx

    private fun audioPadToken(): UInt = config.audioPadToken()

    public fun config(): MultiStreamConfig = config

    public fun userRating(): UInt = userRating

    public fun setUserRating(grade: UInt) {
        userRating = grade
    }

    private fun applyRepetitionPenalty(logits: Tensor): Tensor {
        val rp = repetitionPenalty ?: return logits
        if (rp.penalty == 1.0f) return logits
        val contextSize = rp.contextSize
        val penalty = rp.penalty
        val dev = logits.device()
        val logitsVec = logits.toDtype(DType.F32).toVec1F32()
        val alreadySeen = HashSet<UInt>()
        var nonPadTokens = 0
        val currentTextTokens = textTokens(false)
        for (idx in currentTextTokens.indices.reversed()) {
            val tokenId = currentTextTokens[idx]
            if (tokenId == config.textPadToken ||
                tokenId == config.textEopToken ||
                tokenId == config.textStartToken
            ) {
                continue
            }
            if (nonPadTokens >= contextSize) {
                break
            }
            nonPadTokens += 1
            if (!alreadySeen.add(tokenId)) {
                continue
            }
            val targetIdx = tokenId.toInt()
            if (targetIdx in logitsVec.indices) {
                val cur = logitsVec[targetIdx]
                logitsVec[targetIdx] = if (cur >= 0.0f) cur / penalty else cur * penalty
            }
        }
        return Tensor.fromSlice(logitsVec, Shape(logitsVec.size), dev)
    }

    public fun stepInternal(
        textToken: UInt?,
        inputAudioTokens: UIntArray,
        forceTextToken: UInt?,
        caSrc: CaSrc?,
        conditions: Condition?,
    ): UInt {
        val totalCodebooks = config.totalAudioCodebooks()
        val codes = ArrayList<Tensor?>(totalCodebooks)
        val dev = model.device()
        for (cIdx in inputAudioTokens.indices) {
            audioTokens[stepIdx][cIdx + config.generatedAudioCodebooks] = inputAudioTokens[cIdx]
        }
        val batchSize = if (cfgAlpha != null) 2 else 1
        for (codebook in 0 until totalCodebooks) {
            val t =
                if (codebook == 0 || codebook == config.generatedAudioCodebooks) {
                    if (stepIdx == 0) {
                        audioPadToken()
                    } else {
                        audioTokens[stepIdx - 1][codebook]
                    }
                } else if (stepIdx <= config.acousticDelay) {
                    audioPadToken()
                } else {
                    audioTokens[stepIdx - config.acousticDelay - 1][codebook]
                }
            if (t == UNGENERATED) {
                bail("internal error, ungenerated $stepIdx $codebook")
            }
            val tTensor = Tensor.fromSlice(UIntArray(batchSize) { t }, Shape(batchSize, 1), dev)
            codes.add(tTensor)
        }
        val textTokenTensor =
            textToken?.let { tt ->
                Tensor.fromSlice(UIntArray(batchSize) { tt }, Shape(batchSize, 1), dev)
            }
        val (rawTextLogits, ys) =
            if (caSrc == null) {
                val out = model.forwardCond(textTokenTensor, codes, conditions, StreamMask.empty())
                val logits =
                    if (cfgAlpha == null) {
                        out.logits.i(0).i(0)
                    } else {
                        when (val bSize = out.logits.dim(0)) {
                            2 -> {
                                val l0 =
                                    out.logits
                                        .i(0)
                                        .i(0)
                                        .affine(cfgAlpha, 0.0)
                                val l1 =
                                    out.logits
                                        .i(1)
                                        .i(0)
                                        .affine(cfgAlpha - 1.0, 0.0)
                                l0.sub(l1)
                            }

                            else -> {
                                bail("unexpected batch size $bSize")
                            }
                        }
                    }
                logits to out.hiddenStates
            } else {
                if (cfgAlpha != null) {
                    bail("cfg is not supported with cross attention")
                }
                val out = model.forwardCa(textTokenTensor, codes, caSrc, null, StreamMask.empty())
                out.logits.i(0).i(0) to out.hiddenStates
            }
        val penalizedLogits = applyRepetitionPenalty(rawTextLogits)
        val sampledTextToken =
            forceTextToken
                ?: textLp.sampleF(penalizedLogits) { prs ->
                    if (padMult != null) {
                        prs[config.textPadToken.toInt()] *= exp(padMult)
                    }
                }
        textTokens[stepIdx] = sampledTextToken
        val lastAudioTokens =
            if (cfgAlpha == null) {
                model.depformerSample(
                    xs = ys,
                    textToken = sampledTextToken,
                    forcedAudioTokens = forcedAudioTokens.forcedTokens(stepIdx),
                    lp = audioLp,
                )
            } else {
                model.depformerSampleCfg(
                    xs = ys,
                    cfgAlpha = cfgAlpha,
                    textToken = sampledTextToken,
                    forcedAudioTokens = forcedAudioTokens.forcedTokens(stepIdx),
                    lp = audioLp,
                )
            }
        val padTok = audioPadToken()
        for (cIdx in 0 until config.generatedAudioCodebooks) {
            val delay =
                if (cIdx == 0 || cIdx == config.generatedAudioCodebooks) {
                    0
                } else {
                    config.acousticDelay
                }
            val targetStep = max(0, stepIdx - delay)
            audioTokens[targetStep][cIdx] = lastAudioTokens?.get(cIdx) ?: padTok
        }
        stepIdx += 1
        if (stepIdx >= audioTokens.size) {
            bail("max step-idx reached")
        }
        return sampledTextToken
    }

    public fun stepWithoutCaSrc(
        textToken: UInt,
        inputAudioTokens: UIntArray,
        forceTextToken: UInt?,
    ): UInt = stepInternal(textToken, inputAudioTokens, forceTextToken, null, null)

    public fun step(
        textToken: UInt,
        inputAudioTokens: UIntArray,
        forceTextToken: UInt?,
        caSrc: CaSrc?,
    ): UInt = stepInternal(textToken, inputAudioTokens, forceTextToken, caSrc, null)

    public fun audioTokens(includeAll: Boolean): List<List<UInt>> {
        val maxIdx = if (includeAll) audioTokens.size else min(stepIdx, audioTokens.size)
        return List(maxIdx) { idx -> audioTokens[idx].toList() }
    }

    public fun textTokens(includeAll: Boolean): UIntArray {
        val maxIdx = if (includeAll) textTokens.size else min(stepIdx, textTokens.size)
        return textTokens.copyOfRange(0, maxIdx)
    }

    public fun lastAudioTokens(): UIntArray? {
        if (stepIdx <= config.acousticDelay) {
            return null
        }
        val row = audioTokens[stepIdx - config.acousticDelay - 1]
        val limit = (config.audioVocabSize - 1).toUInt()
        if (row.any { v -> v >= limit }) {
            return null
        }
        return row.copyOf()
    }
}
