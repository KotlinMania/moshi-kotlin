// port-lint: source lm_generate.rs
package io.github.kotlinmania.moshi

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

private val LM_GENERATE_CONFIG_FIELDS: List<String> =
    listOf("audio_delays", "audio_vocab_size", "text_pad_token", "text_eop_token", "text_start_token")

public sealed class Token {
    public data class Set(
        public val token: UInt,
    ) : Token()

    public data object Ungenerated : Token()

    public data object LiteralZero : Token()
}

public data class LmGenerateConfig(
    public val audioDelays: List<Int>,
    public val audioVocabSize: Int,
    public val textPadToken: UInt,
    public val textEopToken: UInt,
    public val textStartToken: UInt,
) : Serialize {
    public fun audioPadToken(): UInt = (audioVocabSize - 1).toUInt()

    public fun audioCodebooks(): Int = audioDelays.size

    public fun maxAudioDelay(): Int = audioDelays.maxOrNull() ?: 0

    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
        serdeCatching {
            val state = serializer.serializeStruct("Config", 5).getOrThrow()
            state.serializeField("audio_delays", audioDelays.asListSerialize { it.asSerialize() }).getOrThrow()
            state.serializeField("audio_vocab_size", audioVocabSize.asSerialize()).getOrThrow()
            state.serializeField("text_pad_token", textPadToken.asSerialize()).getOrThrow()
            state.serializeField("text_eop_token", textEopToken.asSerialize()).getOrThrow()
            state.serializeField("text_start_token", textStartToken.asSerialize()).getOrThrow()
            state.end().getOrThrow()
        }

    public companion object : Deserialize<LmGenerateConfig> {
        override fun <D> deserialize(deserializer: D): SerdeResult<LmGenerateConfig>
            where D : Deserializer =
            deserializer.deserializeStruct(
                "Config",
                LM_GENERATE_CONFIG_FIELDS,
                object : Visitor<LmGenerateConfig> {
                    override fun expecting(): String = "struct lm_generate::Config"

                    override fun <A> visitMap(access: A): SerdeResult<LmGenerateConfig>
                        where A : MapAccess =
                        serdeCatching {
                            var audioDelays: List<Int>? = null
                            var audioVocabSize: Int? = null
                            var textPadToken: UInt? = null
                            var textEopToken: UInt? = null
                            var textStartToken: UInt? = null
                            val i32ListDe = I32Deserialize.asListDeserialize()
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "audio_delays" -> audioDelays = access.nextValueSeed(SeedOf(i32ListDe)).getOrThrow()
                                    "audio_vocab_size" -> audioVocabSize = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "text_pad_token" -> textPadToken = access.nextValueSeed(SeedOf(U32Deserialize)).getOrThrow()
                                    "text_eop_token" -> textEopToken = access.nextValueSeed(SeedOf(U32Deserialize)).getOrThrow()
                                    "text_start_token" -> textStartToken = access.nextValueSeed(SeedOf(U32Deserialize)).getOrThrow()
                                    else -> throw SerdeException(SerdeError.unknownField(key, LM_GENERATE_CONFIG_FIELDS))
                                }
                            }
                            LmGenerateConfig(
                                audioDelays = audioDelays ?: throw SerdeException(SerdeError.missingField("audio_delays")),
                                audioVocabSize = audioVocabSize ?: throw SerdeException(SerdeError.missingField("audio_vocab_size")),
                                textPadToken = textPadToken ?: throw SerdeException(SerdeError.missingField("text_pad_token")),
                                textEopToken = textEopToken ?: throw SerdeException(SerdeError.missingField("text_eop_token")),
                                textStartToken = textStartToken ?: throw SerdeException(SerdeError.missingField("text_start_token")),
                            )
                        }
                },
            )
    }
}

public class LmGenerateState private constructor(
    private val model: LmModel,
    private val audioTokens: List<MutableList<Token>>,
    private val textTokens: MutableList<Token>,
    private val audioLp: LogitsProcessor,
    private val textLp: LogitsProcessor,
    private var stepIdx: Int,
    private val config: LmGenerateConfig,
) {
    public companion object {
        public fun new(
            model: LmModel,
            maxStepIdx: Int,
            audioLp: LogitsProcessor,
            textLp: LogitsProcessor,
            config: LmGenerateConfig,
        ): LmGenerateState {
            val totalLen = maxStepIdx + config.maxAudioDelay()
            val numCodebooks = config.audioCodebooks()
            val audioTokens =
                List(totalLen) {
                    MutableList<Token>(numCodebooks) { Token.Ungenerated }
                }
            val textTokens = MutableList<Token>(totalLen) { Token.Ungenerated }
            return LmGenerateState(
                model = model,
                audioTokens = audioTokens,
                textTokens = textTokens,
                audioLp = audioLp,
                textLp = textLp,
                stepIdx = 0,
                config = config,
            )
        }
    }

    public fun stepIdx(): Int = stepIdx

    public fun audioPadToken(): UInt = config.audioPadToken()

    public fun config(): LmGenerateConfig = config

    public fun setAudioTokens(tokens: List<Token?>) {
        val currentRow = audioTokens[stepIdx]
        val limit = minOf(currentRow.size, tokens.size)
        for (i in 0 until limit) {
            val at = tokens[i]
            if (at != null) {
                currentRow[i] = at
            }
        }
    }

    public fun step(conditions: Condition?) {
        val dev = model.device()
        val numCodebooks = config.audioCodebooks()

        val forcedAudioTokens = ArrayList<UInt?>(numCodebooks)
        for ((codebook, delay) in config.audioDelays.withIndex()) {
            val forcedToken =
                if (stepIdx < delay) {
                    audioPadToken()
                } else {
                    when (val tok = audioTokens[stepIdx - delay][codebook]) {
                        is Token.Ungenerated, is Token.LiteralZero -> null
                        is Token.Set -> tok.token
                    }
                }
            forcedAudioTokens.add(forcedToken)
        }

        val codes = ArrayList<Tensor?>(numCodebooks)
        for ((codebook, delay) in config.audioDelays.withIndex()) {
            val t =
                if (stepIdx <= delay) {
                    audioPadToken()
                } else {
                    when (val tok = audioTokens[stepIdx - delay - 1][codebook]) {
                        is Token.LiteralZero -> null
                        is Token.Set -> tok.token
                        is Token.Ungenerated -> bail("internal error, ungenerated $stepIdx $codebook")
                    }
                }
            val tensor = t?.let { v -> Tensor.fromSlice(uintArrayOf(v), Shape(1, 1), dev) }
            codes.add(tensor)
        }

        val prevTextToken =
            if (stepIdx == 0) {
                config.textStartToken
            } else {
                when (val tok = textTokens[stepIdx - 1]) {
                    is Token.LiteralZero -> null
                    is Token.Set -> tok.token
                    is Token.Ungenerated -> bail("internal error, ungenerated $stepIdx text")
                }
            }
        val textTokenTensor = prevTextToken?.let { v -> Tensor.fromSlice(uintArrayOf(v), Shape(1, 1), dev) }
        val forwardOut = model.forwardCond(textTokenTensor, codes, conditions, StreamMask.empty())
        val textLogits = forwardOut.logits
        val ys = forwardOut.hiddenStates

        val sampledTextToken =
            when (val tok = textTokens[stepIdx]) {
                is Token.Ungenerated -> {
                    val t = textLp.sample(textLogits.i(0).i(0))
                    textTokens[stepIdx] = Token.Set(t)
                    t
                }

                is Token.Set -> {
                    tok.token
                }

                is Token.LiteralZero -> {
                    null
                }
            }

        val sampledAudioTokens =
            model.depformerSample(
                xs = ys,
                textToken = sampledTextToken,
                forcedAudioTokens = forcedAudioTokens,
                lp = audioLp,
            )
        if (sampledAudioTokens != null) {
            for (codebook in sampledAudioTokens.indices) {
                val delay = config.audioDelays[codebook]
                if (stepIdx < delay) {
                    continue
                }
                val row = audioTokens[stepIdx - delay]
                if (row[codebook] == Token.Ungenerated) {
                    row[codebook] = Token.Set(sampledAudioTokens[codebook])
                }
            }
        }
        stepIdx += 1
        if (stepIdx >= audioTokens.size) {
            bail("max step-idx reached")
        }
    }

    public fun lastTextToken(): UInt? =
        if (stepIdx == 0) {
            null
        } else {
            when (val tok = textTokens[stepIdx - 1]) {
                is Token.Set -> tok.token
                is Token.LiteralZero -> null
                is Token.Ungenerated -> bail("internal error, ungenerated step $stepIdx, text")
            }
        }

    public fun lastAudioTokens(): UIntArray? {
        val maxAudioDelay = config.maxAudioDelay()
        if (stepIdx <= maxAudioDelay) {
            return null
        }
        val row = audioTokens[stepIdx - maxAudioDelay - 1]
        val result = UIntArray(row.size)
        for ((cb, audioToken) in row.withIndex()) {
            when (audioToken) {
                is Token.LiteralZero -> return null
                is Token.Set -> result[cb] = audioToken.token
                is Token.Ungenerated -> bail("internal error, ungenerated step $stepIdx, cb $cb")
            }
        }
        return result
    }
}
