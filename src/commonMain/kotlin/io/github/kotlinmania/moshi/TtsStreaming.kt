// port-lint: source tts_streaming.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candle.i
import io.github.kotlinmania.candlenn.Linear
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candlenn.linearNoBias
import io.github.kotlinmania.candletransformers.generation.LogitsProcessor
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.math.max
import kotlin.math.min

@Serializable
public data class TtsStreamingConfig(
    @SerialName("acoustic_delay")
    public val acousticDelay: Int,
    @SerialName("text_pad_token")
    public val textPadToken: UInt,
    @SerialName("text_bos_token")
    public val textBosToken: UInt,
    @SerialName("text_eos_token")
    public val textEosToken: UInt,
    @SerialName("text_eop_token")
    public val textEopToken: UInt,
    @SerialName("text_start_token")
    public val textStartToken: UInt,
    @SerialName("text_audio_delay_in_tokens")
    public val textAudioDelayInTokens: Int,
    @SerialName("max_consecutive_pads")
    public val maxConsecutivePads: Int,
    @SerialName("extra_steps")
    public val extraSteps: Int,
    @SerialName("speaker_cond_duration_s")
    public val speakerCondDurationS: Double,
    @SerialName("speaker_cond_dim")
    public val speakerCondDim: Int,
    @SerialName("speaker_cond_n_speakers")
    public val speakerCondNSpeakers: Int,
) {
    public companion object {
        public fun v202501(): TtsStreamingConfig =
            TtsStreamingConfig(
                acousticDelay = 2,
                textEopToken = 0u,
                textBosToken = 1u,
                textEosToken = 2u,
                textPadToken = 3u,
                textStartToken = 8000u,
                textAudioDelayInTokens = 25,
                maxConsecutivePads = 10,
                extraSteps = 5,
                speakerCondDurationS = 10.0,
                speakerCondDim = 2048,
                speakerCondNSpeakers = 5,
            )
    }
}

public sealed class AllowedTokens {
    public data class Text(
        public val token: UInt,
    ) : AllowedTokens()

    public data object Pad : AllowedTokens()

    public data object PadOrEpad : AllowedTokens()
}

public class TtsStreamingState private constructor(
    private val model: LmModel,
    private val caSrc: CaSrc?,
    private val audioTokensBuffer: MutableList<MutableList<UInt>>,
    private val textTokensBuffer: MutableList<UInt>,
    private var consecutivePads: Int,
    private val audioLp: LogitsProcessor,
    private val textLp: LogitsProcessor,
    private var stepIdxVal: Int,
    private val forcedAudioTokens: ForcedAudioTokens,
    private val cfgAlpha: Double?,
    private val configVal: TtsStreamingConfig,
) {
    public companion object {
        public fun new(
            model: LmModel,
            caSrc: CaSrc?,
            maxStepIdx: Int,
            audioLp: LogitsProcessor,
            textLp: LogitsProcessor,
            cfgAlpha: Double?,
            config: TtsStreamingConfig,
        ): TtsStreamingState {
            val totalSteps = maxStepIdx + config.acousticDelay
            val codebooks = model.generatedAudioCodebooks()
            val audioTokensBuffer =
                MutableList(totalSteps) {
                    MutableList(codebooks) { UNGENERATED }
                }
            val textTokensBuffer = MutableList(totalSteps) { UNGENERATED }
            val forcedAudioTokens =
                ForcedAudioTokens.new(
                    acousticDelay = config.acousticDelay,
                    audioPadToken = model.audioPadToken(),
                    streamCodebooks = intArrayOf(codebooks),
                )
            return TtsStreamingState(
                model = model,
                caSrc = caSrc,
                audioTokensBuffer = audioTokensBuffer,
                textTokensBuffer = textTokensBuffer,
                consecutivePads = 0,
                audioLp = audioLp,
                textLp = textLp,
                stepIdxVal = 0,
                forcedAudioTokens = forcedAudioTokens,
                cfgAlpha = cfgAlpha,
                configVal = config,
            )
        }
    }

    public fun stepIdx(): Int = stepIdxVal

    private fun audioPadToken(): UInt = model.audioPadToken()

    public fun config(): TtsStreamingConfig = configVal

    // The acoustic tokens are written with a delay, so this can create gaps of UNGENERATED
    // tokens in the case where we call stepAudioPrompt after step.
    public fun step(
        prevTextToken: UInt,
        allowedTokens: AllowedTokens,
        conditions: Condition?,
    ): UInt {
        val codebooks = model.generatedAudioCodebooks()
        val codes = ArrayList<Tensor?>(codebooks)
        val dev = model.device()
        val batchSize = if (cfgAlpha != null) 2 else 1
        for (codebook in 0 until codebooks) {
            val t =
                if (codebook == 0) {
                    if (stepIdxVal == 0) {
                        audioPadToken()
                    } else if (stepIdxVal <= configVal.textAudioDelayInTokens) {
                        // The delayed pattern for TTS is a bit special, the audio-pad tokens are used
                        // in the same way as usual, i.e. for the first slice and until the acoustic
                        // delay for semantic tokens.
                        // However for the first couple seconds (set by textAudioDelayInTokens),
                        // the tokens that are not audio-pad are replaced by literal zeros.
                        null
                    } else {
                        audioTokensBuffer[stepIdxVal - 1][codebook]
                    }
                } else if (stepIdxVal <= configVal.acousticDelay) {
                    audioPadToken()
                } else if (stepIdxVal <= configVal.textAudioDelayInTokens + configVal.acousticDelay) {
                    // The same comment as above applies here.
                    null
                } else {
                    audioTokensBuffer[stepIdxVal - configVal.acousticDelay - 1][codebook]
                }
            if (t == UNGENERATED) {
                bail("internal error, ungenerated $stepIdxVal")
            }
            val tTensor =
                t?.let { tokenVal ->
                    Tensor.fromSlice(UIntArray(batchSize) { tokenVal }, Shape(batchSize, 1), dev)
                }
            codes.add(tTensor)
        }
        val prevTextTensor =
            Tensor.fromSlice(UIntArray(batchSize) { prevTextToken }, Shape(batchSize, 1), dev)
        val forwardOut =
            if (caSrc == null) {
                model.forwardCond(prevTextTensor, codes, conditions, StreamMask.empty())
            } else {
                model.forwardCa(prevTextTensor, codes, caSrc, conditions, StreamMask.empty())
            }
        val rawTextLogits = forwardOut.logits
        val ys = forwardOut.hiddenStates
        val textLogits =
            if (cfgAlpha == null) {
                rawTextLogits.i(0).i(0)
            } else {
                when (val bSize = rawTextLogits.dim(0)) {
                    2 -> {
                        val l0 = rawTextLogits.i(0).i(0).affine(cfgAlpha, 0.0)
                        val l1 = rawTextLogits.i(1).i(0).affine(cfgAlpha - 1.0, 0.0)
                        l0.sub(l1)
                    }

                    else -> {
                        bail("unexpected batch size $bSize")
                    }
                }
            }
        // When in tts mode, there are only two possible outcomes corresponding to tokens 0 and 3.
        // 0 -> EOP or the next text token, this is ambiguous, a list of consecutive 0s correspond to
        //   word + EOP + word + EOP ...
        // 3 -> pad.
        // This will change when the simplerleaver lands.
        val textToken =
            when (allowedTokens) {
                is AllowedTokens.Text -> {
                    allowedTokens.token
                }

                is AllowedTokens.Pad -> {
                    configVal.textPadToken
                }

                is AllowedTokens.PadOrEpad -> {
                    if (consecutivePads > configVal.maxConsecutivePads) {
                        configVal.textEopToken
                    } else {
                        val sampled = textLp.sample(textLogits)
                        if (sampled == configVal.textPadToken) {
                            configVal.textPadToken
                        } else {
                            configVal.textEopToken
                        }
                    }
                }
            }
        if (textToken == configVal.textPadToken) {
            consecutivePads += 1
        } else {
            consecutivePads = 0
        }
        textTokensBuffer[stepIdxVal] = textToken
        val lastAudioTokens =
            if (stepIdxVal < configVal.textAudioDelayInTokens) {
                null
            } else if (cfgAlpha == null) {
                model.depformerSample(
                    xs = ys,
                    textToken = textToken,
                    forcedAudioTokens = forcedAudioTokens.forcedTokens(stepIdxVal),
                    lp = audioLp,
                )
            } else {
                model.depformerSampleCfg(
                    xs = ys,
                    cfgAlpha = cfgAlpha,
                    textToken = textToken,
                    forcedAudioTokens = forcedAudioTokens.forcedTokens(stepIdxVal),
                    lp = audioLp,
                )
            }
        val padToken = audioPadToken()
        for (cIdx in 0 until codebooks) {
            val delay = if (cIdx == 0) 0 else configVal.acousticDelay
            val targetStep = max(0, stepIdxVal - delay)
            if (audioTokensBuffer[targetStep][cIdx] == UNGENERATED) {
                audioTokensBuffer[targetStep][cIdx] =
                    lastAudioTokens?.get(cIdx) ?: padToken
            }
        }
        stepIdxVal += 1
        if (stepIdxVal >= audioTokensBuffer.size) {
            bail("max step-idx reached")
        }
        return textToken
    }

    public fun overwriteLastTextToken(textToken: UInt) {
        if (stepIdxVal == 0) {
            bail("cannot overwrite first token")
        }
        if (textToken == UNGENERATED) {
            bail("cannot overwrite with UNGENERATED")
        }
        textTokensBuffer[stepIdxVal - 1] = textToken
    }

    /**
     * If includeAll is set, all the time steps are returned. Otherwise only the timesteps that
     * have been generated are handled.
     */
    public fun audioTokens(includeAll: Boolean): List<List<UInt>> {
        val slice =
            if (includeAll) {
                audioTokensBuffer
            } else {
                val maxIdx = min(stepIdxVal, audioTokensBuffer.size)
                audioTokensBuffer.subList(0, maxIdx)
            }
        return slice.map { it.toList() }
    }

    public fun textTokens(includeAll: Boolean): List<UInt> =
        if (includeAll) {
            textTokensBuffer.toList()
        } else {
            val maxIdx = min(stepIdxVal, textTokensBuffer.size)
            textTokensBuffer.subList(0, maxIdx).toList()
        }

    public fun lastAudioTokens(): List<UInt>? =
        if (stepIdxVal <= configVal.acousticDelay) {
            null
        } else {
            // stepIdx is in advance by 1 + there is a 2 token delay on audio tokens.
            val tokens = audioTokensBuffer[stepIdxVal - configVal.acousticDelay - 1]
            if (tokens.any { v -> v >= audioPadToken() }) {
                null
            } else {
                tokens.toList()
            }
        }

    public fun audioCodebooks(): Int = model.generatedAudioCodebooks()

    public fun device(): Device = model.device()

    public fun dtype(): DType = model.dtype()
}

public enum class Speaker {
    Main,
    Other,
}

public fun interface PromptTokenizer {
    public fun encode(word: String): List<UInt>
}

public data class PromptSegment(
    public val tokens: List<UInt>,
    public val speaker: Speaker,
)

public fun tokenizePrompt(
    text: List<String>,
    textBosToken: UInt,
    textEosToken: UInt,
    encode: PromptTokenizer,
): List<PromptSegment> {
    val prompt = mutableListOf<PromptSegment>()
    for ((turnIdx, turn) in text.withIndex()) {
        val (speaker, turnToken) =
            if (turnIdx % 2 == 0) {
                Speaker.Main to textBosToken
            } else {
                Speaker.Other to textEosToken
            }
        for ((wordIdx, word) in turn.split(' ').withIndex()) {
            val encodedWord = encode.encode(word).toMutableList()
            if (wordIdx == 0 && speaker == Speaker.Main) {
                encodedWord.add(0, turnToken)
            }
            if (encodedWord.isNotEmpty()) {
                prompt.add(PromptSegment(encodedWord, speaker))
            }
        }
    }
    return prompt
}

public class SpeakerEncoder(
    private val mimi: Mimi,
    private val learntPadding: Tensor,
    private val proj: Linear,
    private val nSpeakers: Int,
    private val condDim: Int,
    private val deviceVal: Device,
    private val dtype: DType,
) {
    public companion object {
        public fun new(
            mimi: Mimi,
            speakerCondDim: Int,
            speakerCondNSpeakers: Int,
            dtype: DType,
            vb: VarBuilder,
        ): SpeakerEncoder {
            val learntPadding =
                vb.get(
                    Shape(1, 1, speakerCondDim),
                    "condition_provider.conditioners.speaker_wavs.learnt_padding",
                )
            val mimiDim = mimi.config().seanet.dimension
            val proj =
                linearNoBias(
                    mimiDim,
                    speakerCondDim,
                    vb.pp("condition_provider.conditioners.speaker_wavs.output_proj"),
                )
            return SpeakerEncoder(
                mimi = mimi,
                learntPadding = learntPadding,
                proj = proj,
                nSpeakers = speakerCondNSpeakers,
                condDim = speakerCondDim,
                deviceVal = vb.device(),
                dtype = dtype,
            )
        }
    }

    public fun device(): Device = deviceVal

    public fun sampleRate(): Double = mimi.config().sampleRate

    public fun encode(speakers: List<Tensor>): Tensor {
        if (speakers.isEmpty()) {
            bail("empty speakers in encode")
        }
        val pcms = mutableListOf<Tensor>()
        for (pcm in speakers.take(nSpeakers)) {
            val stdev =
                pcm
                    .broadcastSub(pcm.meanAll())
                    .sqr()
                    .meanAll()
                    .sqrt()
            val normPcm = pcm.affine(0.08, 0.0).broadcastDiv(stdev)
            pcms.add(normPcm)
        }
        val numSpeakers = pcms.size
        val pcmCat = Tensor.cat(pcms, 0)
        mimi.resetState()
        var embeddings =
            mimi
                .encodePreQuantize(pcmCat)
                .t()
                .apply(proj)
        if (numSpeakers < nSpeakers) {
            val lp =
                embeddings
                    .narrow(0, 0, 1)
                    .zerosLike()
                    .broadcastAdd(learntPadding)
            val embs = mutableListOf(embeddings)
            while (embs.size < nSpeakers - numSpeakers + 1) {
                embs.add(lp)
            }
            embeddings = Tensor.cat(embs, 0)
        }
        val flatEmbeddings = embeddings.flatten(0, 1).unsqueeze(0)
        val sinEmbeddings = addSinEmbeddings(flatEmbeddings)
        return sinEmbeddings.toDtype(dtype)
    }

    public fun empty(): Tensor {
        val embeddings =
            learntPadding.broadcastAs(Shape(1, nSpeakers * 125, condDim))
        val sinEmbeddings = addSinEmbeddings(embeddings)
        return sinEmbeddings.toDtype(dtype)
    }
}
