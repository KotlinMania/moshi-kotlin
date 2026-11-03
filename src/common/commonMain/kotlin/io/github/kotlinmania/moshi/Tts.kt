// port-lint: source tts.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candlenn.Linear
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candlenn.linearNoBias
import io.github.kotlinmania.candletransformers.generation.LogitsProcessor
import io.github.kotlinmania.candletransformers.generation.Sampling
import io.github.kotlinmania.candletransformers.models.T5Config
import io.github.kotlinmania.candletransformers.models.T5EncoderModel
import kotlin.math.max
import kotlin.math.pow

public data class TtsConfig(
    public val t5: T5Config,
    public val lm: LmConfig,
    public val mimi: MimiConfig,
    public val maxDurationS: Double,
    public val speakerCondDurationS: Double,
    public val maxSpeakers: Int,
) {
    public companion object {
        public fun v01(t5: T5Config): TtsConfig {
            val lm = LmConfig.ttsV01()
            val mimi = MimiConfig.v01(null)
            return TtsConfig(
                t5 = t5,
                lm = lm,
                mimi = mimi,
                maxDurationS = 60.0,
                speakerCondDurationS = 4.0,
                maxSpeakers = 5,
            )
        }

        public fun v02(t5: T5Config): TtsConfig {
            val lm = LmConfig.ttsV01()
            val mimi = MimiConfig.v01(null)
            return TtsConfig(
                t5 = t5,
                lm = lm,
                mimi = mimi,
                maxDurationS = 60.0,
                speakerCondDurationS = 10.0,
                maxSpeakers = 2,
            )
        }
    }
}

public class SpeakerConditionerBlock(
    public val mimi: Mimi,
    public val proj: Linear,
)

public class TtsModel(
    private val t5: T5EncoderModel,
    public val lm: LmModel,
    private val speakerCond: SpeakerConditionerBlock?,
    private val t5Proj: Linear,
    public val sampleRate: Double,
    private val frameRate: Double,
    private val audioVocabSize: UInt,
    private val audioCodebooks: Int,
    public val maxDurationS: Double,
    private val maxSpeakers: Int,
    private var endOfGen: Int? = null,
) {
    public companion object {
        public fun new(
            cfg: TtsConfig,
            vbT5: VarBuilder,
            vbLm: VarBuilder,
            vbSpeakerCond: VarBuilder?,
        ): TtsModel {
            val t5 = T5EncoderModel.load(vbT5, cfg.t5)
            val speakerCond =
                if (vbSpeakerCond == null) {
                    null
                } else {
                    val mimi = Mimi.new(cfg.mimi.copy(), vbSpeakerCond)
                    val proj =
                        linearNoBias(
                            cfg.mimi.seanet.dimension,
                            cfg.lm.transformer.dModel,
                            vbLm.pp("condition_provider.conditioners.speaker_wavs.output_proj"),
                        )
                    SpeakerConditionerBlock(mimi, proj)
                }
            val t5ProjName =
                if (speakerCond != null) {
                    "condition_provider.conditioners.diarized_transcript_in_segment.output_proj"
                } else {
                    "condition_provider.conditioners.transcript_in_segment.output_proj"
                }
            val t5Proj =
                linearNoBias(
                    cfg.t5.dModel,
                    cfg.lm.transformer.dModel,
                    vbLm.pp(t5ProjName),
                )
            val lm = LmModel.new(cfg.lm, MaybeQuantizedVarBuilder.Real(vbLm))
            return TtsModel(
                t5 = t5,
                lm = lm,
                speakerCond = speakerCond,
                t5Proj = t5Proj,
                sampleRate = cfg.mimi.sampleRate,
                frameRate = cfg.mimi.frameRate,
                audioVocabSize = cfg.lm.audioVocabSize.toUInt(),
                audioCodebooks = cfg.lm.audioCodebooks,
                maxDurationS = cfg.maxDurationS,
                maxSpeakers = cfg.maxSpeakers,
                endOfGen = null,
            )
        }
    }

    public fun conditions(
        tokenIds: Tensor,
        speakerPcm: Tensor?,
    ): Tensor {
        val t5Condition = t5.forward(tokenIds).toDtype(DType.BF16).apply(t5Proj)
        return if (speakerPcm == null) {
            t5Condition
        } else {
            val scBlock =
                speakerCond
                    ?: bail("speakerPcm specified without a speaker-cond model")
            val sc =
                scBlock.mimi
                    .encodePreQuantize(speakerPcm)
                    .t()
                    .toDtype(DType.BF16)
                    .apply(scBlock.proj)
            val z = sc.zerosLike()
            val c1List = mutableListOf(t5Condition, sc)
            val c2List = mutableListOf(t5Condition, z)
            repeat(maxSpeakers - 1) {
                c1List.add(z)
                c2List.add(z)
            }
            val c1 = Tensor.cat(c1List, 1)
            val c2 = Tensor.cat(c2List, 1)
            val xs = Tensor.cat(listOf(c1, c2), 0)
            addSinEmbeddings(xs)
        }
    }

    public fun sample(
        conditions: Tensor,
        cfgAlpha: Double,
    ): List<List<UInt>> {
        val lp =
            LogitsProcessor.fromSampling(
                299792458uL,
                Sampling.TopK(k = 100, temperature = 0.8),
            )
        return sampleLp(conditions, cfgAlpha, lp)
    }

    public fun sampleLp(
        conditions: Tensor,
        cfgAlpha: Double,
        lp: LogitsProcessor,
    ): List<List<UInt>> {
        val maxSteps = (maxDurationS * frameRate).toInt() + 1
        val audioTokens =
            MutableList(maxSteps + 2) {
                MutableList(audioCodebooks) { UInt.MAX_VALUE }
            }
        val forcedAudioTokens =
            ForcedAudioTokens.new(
                acousticDelay = 2,
                audioPadToken = lm.audioPadToken(),
                streamCodebooks = intArrayOf(audioCodebooks),
            )
        val quantizerBins = audioVocabSize - 2u
        for (stepIdx in 0 until (maxSteps + 2)) {
            val codes = ArrayList<Tensor?>(audioCodebooks)
            for (codebook in 0 until audioCodebooks) {
                val t =
                    if (codebook == 0) {
                        if (stepIdx == 0) {
                            audioVocabSize - 1u
                        } else {
                            audioTokens[stepIdx - 1][0]
                        }
                    } else if (stepIdx <= 2) {
                        audioVocabSize - 1u
                    } else {
                        audioTokens[stepIdx - 3][codebook]
                    }
                val tTensor = Tensor.fromSlice(uintArrayOf(t), Shape(1, 1), conditions.device())
                codes.add(tTensor)
            }
            val forwardOut =
                lm.forwardCa(
                    textIds = null,
                    audioIds = codes,
                    caSrc = CaSrc.Tokens(conditions),
                    conditions = null,
                    mask = StreamMask.empty(),
                )
            val ys = forwardOut.hiddenStates
            val lastAudioTokens =
                if (speakerCond != null) {
                    lm.depformerSampleCfg(
                        xs = ys,
                        cfgAlpha = cfgAlpha,
                        textToken = null,
                        forcedAudioTokens = forcedAudioTokens.forcedTokens(stepIdx),
                        lp = lp,
                    )
                } else {
                    lm.depformerSample(
                        xs = ys,
                        textToken = null,
                        forcedAudioTokens = forcedAudioTokens.forcedTokens(stepIdx),
                        lp = lp,
                    )
                } ?: bail("no depformer")
            for (cIdx in lastAudioTokens.indices) {
                val token = lastAudioTokens[cIdx]
                if (stepIdx > 0 && token >= quantizerBins && endOfGen == null) {
                    // Continue generating for two steps to get the final acoustic tokens.
                    endOfGen = stepIdx + 2
                }
                val delay = if (cIdx == 0) 0 else 2
                val targetStep = max(0, stepIdx - delay)
                audioTokens[targetStep][cIdx] = token
            }
            if (stepIdx == endOfGen) {
                break
            }
        }
        return audioTokens.map { it.toList() }
    }
}

public fun addSinEmbeddings(xs: Tensor): Tensor {
    val targetDtype = xs.dtype()
    val (_, seqLen, dim) = xs.dims3()
    val dev = xs.device()
    val halfDim = dim / 2
    val positions =
        Tensor
            .arange(0, seqLen, dev)
            .unsqueeze(1)
            .toDtype(DType.F32)
    val invFreq =
        FloatArray(halfDim) { i ->
            1.0f / 10000.0f.pow(i.toFloat() / (halfDim - 1).toFloat())
        }
    val invFreqTensor = Tensor.fromSlice(invFreq, Shape(1, halfDim), dev)
    val freqs = positions.broadcastMul(invFreqTensor)
    val posEmb = Tensor.cat(listOf(freqs.cos(), freqs.sin()), D.Minus1)
    val xsOut = xs.toDtype(DType.F32).broadcastAdd(posEmb)
    return xsOut.toDtype(targetDtype)
}
