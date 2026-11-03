// port-lint: source lm.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.i
import io.github.kotlinmania.candlenn.Activation
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candletransformers.QuantizedVarBuilder
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
import io.github.kotlinmania.serdecore.de.Visitor
import io.github.kotlinmania.serdecore.de.mapDeserialize

private val DEPFORMER_CONFIG_FIELDS: List<String> = listOf("transformer", "num_slices", "low_rank_embeddings")
private val EXTRA_HEADS_CONFIG_FIELDS: List<String> = listOf("num_heads", "dim")
private val LM_CONFIG_FIELDS: List<String> =
    listOf(
        "transformer",
        "depformer",
        "text_in_vocab_size",
        "text_out_vocab_size",
        "audio_vocab_size",
        "audio_codebooks",
        "conditioners",
        "extra_heads",
    )

public data class LmForwardOutput(
    public val logits: Tensor,
    public val hiddenStates: Tensor,
)

public data class DepFormerConfig(
    public val transformer: TransformerConfig,
    public val numSlices: Int,
    public val lowRankEmbeddings: Int? = null,
) {
    public companion object : Deserialize<DepFormerConfig> {
        override fun <D> deserialize(deserializer: D): SerdeResult<DepFormerConfig>
            where D : Deserializer =
            deserializer.deserializeStruct(
                "DepFormerConfig",
                DEPFORMER_CONFIG_FIELDS,
                object : Visitor<DepFormerConfig> {
                    override fun expecting(): String = "struct DepFormerConfig"

                    override fun <A> visitMap(access: A): SerdeResult<DepFormerConfig>
                        where A : MapAccess =
                        serdeCatching {
                            var transformer: TransformerConfig? = null
                            var numSlices: Int? = null
                            var lowRankEmbeddings: Int? = null
                            val optIntDe = I32Deserialize.asNullableDeserialize()
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "transformer" -> transformer = access.nextValueSeed(SeedOf(TransformerConfig)).getOrThrow()
                                    "num_slices" -> numSlices = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "low_rank_embeddings" -> lowRankEmbeddings = access.nextValueSeed(SeedOf(optIntDe)).getOrThrow()
                                    else -> throw SerdeException(SerdeError.unknownField(key, DEPFORMER_CONFIG_FIELDS))
                                }
                            }
                            DepFormerConfig(
                                transformer = transformer ?: throw SerdeException(SerdeError.missingField("transformer")),
                                numSlices = numSlices ?: throw SerdeException(SerdeError.missingField("num_slices")),
                                lowRankEmbeddings = lowRankEmbeddings,
                            )
                        }
                },
            )
    }
}

public data class ExtraHeadsConfig(
    public val numHeads: Int,
    public val dim: Int,
) {
    public companion object : Deserialize<ExtraHeadsConfig> {
        override fun <D> deserialize(deserializer: D): SerdeResult<ExtraHeadsConfig>
            where D : Deserializer =
            deserializer.deserializeStruct(
                "ExtraHeadsConfig",
                EXTRA_HEADS_CONFIG_FIELDS,
                object : Visitor<ExtraHeadsConfig> {
                    override fun expecting(): String = "struct ExtraHeadsConfig"

                    override fun <A> visitMap(access: A): SerdeResult<ExtraHeadsConfig>
                        where A : MapAccess =
                        serdeCatching {
                            var numHeads: Int? = null
                            var dim: Int? = null
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "num_heads" -> numHeads = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "dim" -> dim = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    else -> throw SerdeException(SerdeError.unknownField(key, EXTRA_HEADS_CONFIG_FIELDS))
                                }
                            }
                            ExtraHeadsConfig(
                                numHeads = numHeads ?: throw SerdeException(SerdeError.missingField("num_heads")),
                                dim = dim ?: throw SerdeException(SerdeError.missingField("dim")),
                            )
                        }
                },
            )
    }
}

public data class LmConfig(
    public val transformer: TransformerConfig,
    public val depformer: DepFormerConfig? = null,
    public val textInVocabSize: Int,
    public val textOutVocabSize: Int,
    public val audioVocabSize: Int,
    public val audioCodebooks: Int,
    public val conditioners: kotlin.collections.Map<String, ConditionerConfig>? = null,
    public val extraHeads: ExtraHeadsConfig? = null,
) {
    public companion object : Deserialize<LmConfig> {
        override fun <D> deserialize(deserializer: D): SerdeResult<LmConfig>
            where D : Deserializer =
            deserializer.deserializeStruct(
                "Config",
                LM_CONFIG_FIELDS,
                object : Visitor<LmConfig> {
                    override fun expecting(): String = "struct lm::Config"

                    override fun <A> visitMap(access: A): SerdeResult<LmConfig>
                        where A : MapAccess =
                        serdeCatching {
                            var transformer: TransformerConfig? = null
                            var depformer: DepFormerConfig? = null
                            var textInVocabSize: Int? = null
                            var textOutVocabSize: Int? = null
                            var audioVocabSize: Int? = null
                            var audioCodebooks: Int? = null
                            var conditioners: kotlin.collections.Map<String, ConditionerConfig>? = null
                            var extraHeads: ExtraHeadsConfig? = null
                            val optDepDe = DepFormerConfig.asNullableDeserialize()
                            val optCondDe = mapDeserialize(StringDeserialize, ConditionerConfig).asNullableDeserialize()
                            val optExtraDe = ExtraHeadsConfig.asNullableDeserialize()
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "transformer" -> transformer = access.nextValueSeed(SeedOf(TransformerConfig)).getOrThrow()
                                    "depformer" -> depformer = access.nextValueSeed(SeedOf(optDepDe)).getOrThrow()
                                    "text_in_vocab_size" -> textInVocabSize = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "text_out_vocab_size" -> textOutVocabSize = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "audio_vocab_size" -> audioVocabSize = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "audio_codebooks" -> audioCodebooks = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "conditioners" -> conditioners = access.nextValueSeed(SeedOf(optCondDe)).getOrThrow()
                                    "extra_heads" -> extraHeads = access.nextValueSeed(SeedOf(optExtraDe)).getOrThrow()
                                    else -> throw SerdeException(SerdeError.unknownField(key, LM_CONFIG_FIELDS))
                                }
                            }
                            LmConfig(
                                transformer = transformer ?: throw SerdeException(SerdeError.missingField("transformer")),
                                depformer = depformer,
                                textInVocabSize = textInVocabSize ?: throw SerdeException(SerdeError.missingField("text_in_vocab_size")),
                                textOutVocabSize = textOutVocabSize ?: throw SerdeException(SerdeError.missingField("text_out_vocab_size")),
                                audioVocabSize = audioVocabSize ?: throw SerdeException(SerdeError.missingField("audio_vocab_size")),
                                audioCodebooks = audioCodebooks ?: throw SerdeException(SerdeError.missingField("audio_codebooks")),
                                conditioners = conditioners,
                                extraHeads = extraHeads,
                            )
                        }
                },
            )

        public fun depformerCfg(numSlices: Int): DepFormerConfig {
            val transformerCfg =
                TransformerConfig(
                    dModel = 1024,
                    numHeads = 16,
                    numLayers = 6,
                    dimFeedforward = 1024 * 4,
                    causal = true,
                    normFirst = true,
                    biasFf = false,
                    biasAttn = false,
                    layerScale = null,
                    context = numSlices,
                    maxPeriod = 10000,
                    useConvBlock = false,
                    useConvBias = true,
                    crossAttention = null,
                    gating = Activation.Silu,
                    norm = NormType.RmsNorm,
                    positionalEmbedding = PositionalEmbedding.None,
                    convLayout = false,
                    convKernelSize = 3,
                    kvRepeat = 1,
                    maxSeqLen = 4096,
                    sharedCrossAttn = false,
                )
            return DepFormerConfig(
                transformer = transformerCfg,
                numSlices = numSlices,
                lowRankEmbeddings = null,
            )
        }

        public fun v01(): LmConfig {
            val lmCfg =
                TransformerConfig(
                    dModel = 4096,
                    numHeads = 32,
                    numLayers = 32,
                    dimFeedforward = 4096 * 4,
                    causal = true,
                    normFirst = true,
                    biasFf = false,
                    biasAttn = false,
                    layerScale = null,
                    context = 3000,
                    maxPeriod = 10000,
                    useConvBlock = false,
                    useConvBias = true,
                    crossAttention = null,
                    gating = Activation.Silu,
                    norm = NormType.RmsNorm,
                    positionalEmbedding = PositionalEmbedding.Rope,
                    convLayout = false,
                    convKernelSize = 3,
                    kvRepeat = 1,
                    maxSeqLen = 4096,
                    sharedCrossAttn = false,
                )
            return LmConfig(
                transformer = lmCfg,
                depformer = depformerCfg(8),
                audioVocabSize = 2049,
                textInVocabSize = 32001,
                textOutVocabSize = 32000,
                audioCodebooks = 8,
                conditioners = null,
                extraHeads = null,
            )
        }

        public fun v01Vision(): LmConfig {
            val lmCfg =
                TransformerConfig(
                    dModel = 4096,
                    numHeads = 32,
                    numLayers = 32,
                    dimFeedforward = 4096 * 4,
                    causal = true,
                    normFirst = true,
                    biasFf = false,
                    biasAttn = false,
                    layerScale = null,
                    context = 3000,
                    maxPeriod = 10000,
                    useConvBlock = false,
                    useConvBias = true,
                    crossAttention =
                        CrossAttentionConfig(
                            gating = CrossAttentionGating.ConditionalGatedSigmoid,
                            norm = NormType.RmsNorm,
                            kvInDim = null,
                        ),
                    gating = Activation.Silu,
                    norm = NormType.RmsNorm,
                    positionalEmbedding = PositionalEmbedding.Rope,
                    convLayout = false,
                    convKernelSize = 3,
                    kvRepeat = 1,
                    maxSeqLen = 4096,
                    sharedCrossAttn = true,
                )
            return LmConfig(
                transformer = lmCfg,
                depformer = depformerCfg(8),
                audioVocabSize = 2049,
                textInVocabSize = 32001,
                textOutVocabSize = 32000,
                audioCodebooks = 8,
                conditioners = null,
                extraHeads = null,
            )
        }

        public fun v01VisionStreaming(numSlices: Int): LmConfig {
            val base = v01Vision()
            val updatedDepformer =
                base.depformer?.let { df ->
                    df.copy(
                        numSlices = numSlices,
                        transformer = df.transformer.copy(context = numSlices),
                    )
                }
            return base.copy(audioCodebooks = 16, depformer = updatedDepformer)
        }

        public fun v01Streaming(numSlices: Int): LmConfig {
            val base = v01()
            val updatedDepformer =
                base.depformer?.let { df ->
                    df.copy(
                        numSlices = numSlices,
                        transformer = df.transformer.copy(context = numSlices),
                    )
                }
            return base.copy(audioCodebooks = 16, depformer = updatedDepformer)
        }

        public fun v01Asr(): LmConfig {
            val base = v01()
            val updatedDepformer =
                base.depformer?.let { df ->
                    df.copy(
                        numSlices = 0,
                        transformer = df.transformer.copy(context = 0),
                    )
                }
            return base.copy(audioCodebooks = 8, depformer = updatedDepformer)
        }

        public fun ttsV01(): LmConfig {
            val lmCfg =
                TransformerConfig(
                    dModel = 2048,
                    numHeads = 32,
                    numLayers = 48,
                    dimFeedforward = 4096 * 2,
                    causal = true,
                    normFirst = true,
                    biasFf = false,
                    biasAttn = false,
                    layerScale = null,
                    context = 4096,
                    maxPeriod = 10000,
                    useConvBlock = false,
                    useConvBias = true,
                    crossAttention =
                        CrossAttentionConfig(
                            gating = CrossAttentionGating.Normal,
                            norm = NormType.LayerNorm,
                            kvInDim = null,
                        ),
                    gating = null,
                    norm = NormType.LayerNorm,
                    positionalEmbedding = PositionalEmbedding.Rope,
                    convLayout = false,
                    convKernelSize = 3,
                    kvRepeat = 1,
                    maxSeqLen = 4096,
                    sharedCrossAttn = false,
                )
            return LmConfig(
                transformer = lmCfg,
                depformer = depformerCfg(16),
                audioVocabSize = 2050,
                textInVocabSize = 32001,
                textOutVocabSize = 32001,
                audioCodebooks = 16,
                conditioners = null,
                extraHeads = null,
            )
        }

        public fun s2sV01(): LmConfig {
            val lmCfg =
                TransformerConfig(
                    dModel = 2048,
                    numHeads = 16,
                    numLayers = 16,
                    dimFeedforward = 4096 * 2,
                    causal = true,
                    normFirst = true,
                    biasFf = false,
                    biasAttn = false,
                    layerScale = null,
                    context = 3000,
                    maxPeriod = 10000,
                    useConvBlock = false,
                    useConvBias = true,
                    crossAttention = null,
                    gating = Activation.Silu,
                    norm = NormType.RmsNorm,
                    positionalEmbedding = PositionalEmbedding.Rope,
                    convLayout = false,
                    convKernelSize = 3,
                    kvRepeat = 1,
                    maxSeqLen = 4096,
                    sharedCrossAttn = false,
                )
            return LmConfig(
                transformer = lmCfg,
                depformer = depformerCfg(16),
                audioVocabSize = 2049,
                textInVocabSize = 48001,
                textOutVocabSize = 48000,
                audioCodebooks = 16,
                conditioners = null,
                extraHeads = null,
            )
        }

        public fun s2sV01Streaming(numSlices: Int): LmConfig {
            val base = s2sV01()
            val updatedDepformer =
                base.depformer?.let { df ->
                    df.copy(
                        numSlices = numSlices,
                        transformer = df.transformer.copy(context = numSlices),
                    )
                }
            return base.copy(audioCodebooks = 16, depformer = updatedDepformer)
        }

        public fun asrV011b(): LmConfig {
            val lmCfg =
                TransformerConfig(
                    dModel = 2048,
                    numHeads = 16,
                    numLayers = 16,
                    dimFeedforward = 2048 * 4,
                    causal = true,
                    normFirst = true,
                    biasFf = false,
                    biasAttn = false,
                    layerScale = null,
                    context = 750,
                    maxPeriod = 100_000,
                    useConvBlock = false,
                    useConvBias = true,
                    crossAttention = null,
                    gating = Activation.Silu,
                    norm = NormType.RmsNorm,
                    positionalEmbedding = PositionalEmbedding.Rope,
                    convLayout = false,
                    convKernelSize = 3,
                    kvRepeat = 1,
                    maxSeqLen = 4096,
                    sharedCrossAttn = false,
                )
            return LmConfig(
                transformer = lmCfg,
                depformer = null,
                audioVocabSize = 2049,
                textInVocabSize = 48001,
                textOutVocabSize = 48000,
                audioCodebooks = 8,
                conditioners = null,
                extraHeads = null,
            )
        }

        public fun asr300m202501(): LmConfig {
            val lmCfg =
                TransformerConfig(
                    dModel = 1024,
                    numHeads = 8,
                    numLayers = 16,
                    dimFeedforward = 1024 * 4,
                    causal = true,
                    normFirst = true,
                    biasFf = false,
                    biasAttn = false,
                    layerScale = null,
                    context = 750,
                    maxPeriod = 100_000,
                    useConvBlock = false,
                    useConvBias = true,
                    crossAttention = null,
                    gating = Activation.Silu,
                    norm = NormType.RmsNorm,
                    positionalEmbedding = PositionalEmbedding.Rope,
                    convLayout = false,
                    convKernelSize = 3,
                    kvRepeat = 1,
                    maxSeqLen = 4096,
                    sharedCrossAttn = false,
                )
            return LmConfig(
                transformer = lmCfg,
                depformer = null,
                audioVocabSize = 2049,
                textInVocabSize = 48001,
                textOutVocabSize = 48000,
                audioCodebooks = 32,
                conditioners = null,
                extraHeads = null,
            )
        }

        public fun tts202501(): LmConfig {
            val lmCfg =
                TransformerConfig(
                    dModel = 2048,
                    numHeads = 32,
                    numLayers = 48,
                    dimFeedforward = 2048 * 4,
                    causal = true,
                    normFirst = true,
                    biasFf = false,
                    biasAttn = false,
                    layerScale = null,
                    context = 500,
                    maxPeriod = 10000,
                    useConvBlock = false,
                    useConvBias = true,
                    crossAttention =
                        CrossAttentionConfig(
                            gating = CrossAttentionGating.Normal,
                            norm = NormType.LayerNorm,
                            kvInDim = null,
                        ),
                    gating = Activation.Silu,
                    norm = NormType.RmsNorm,
                    positionalEmbedding = PositionalEmbedding.Rope,
                    convLayout = false,
                    convKernelSize = 3,
                    kvRepeat = 1,
                    maxSeqLen = 4096,
                    sharedCrossAttn = false,
                )
            return LmConfig(
                transformer = lmCfg,
                depformer = depformerCfg(32),
                audioVocabSize = 2049,
                textInVocabSize = 8001,
                textOutVocabSize = 8000,
                audioCodebooks = 32,
                conditioners = null,
                extraHeads = null,
            )
        }

        public fun s2s2b16rvq202501(): LmConfig {
            val lmCfg =
                TransformerConfig(
                    dModel = 2560,
                    numHeads = 20,
                    numLayers = 24,
                    dimFeedforward = 2560 * 4,
                    causal = true,
                    normFirst = true,
                    biasFf = false,
                    biasAttn = false,
                    layerScale = null,
                    context = 3000,
                    maxPeriod = 100000,
                    useConvBlock = false,
                    useConvBias = true,
                    crossAttention = null,
                    gating = Activation.Silu,
                    norm = NormType.RmsNorm,
                    positionalEmbedding = PositionalEmbedding.Rope,
                    convLayout = false,
                    convKernelSize = 3,
                    kvRepeat = 1,
                    maxSeqLen = 4096,
                    sharedCrossAttn = false,
                )
            return LmConfig(
                transformer = lmCfg,
                depformer = depformerCfg(16),
                audioVocabSize = 2049,
                textInVocabSize = 48001,
                textOutVocabSize = 48000,
                audioCodebooks = 32,
                conditioners = null,
                extraHeads = null,
            )
        }
    }
}

internal class LowRankEmbeddings(
    val embeddings: MaybeQuantizedEmbedding,
    val lowRank: MaybeQuantizedLinear?,
) : Module {
    companion object {
        fun new(
            inVocabSize: Int,
            dim: Int,
            lowRankDim: Int?,
            vb: MaybeQuantizedVarBuilder,
        ): LowRankEmbeddings =
            if (lowRankDim == null) {
                val embeddings = MaybeQuantizedEmbedding.new(inVocabSize, dim, vb)
                LowRankEmbeddings(embeddings, null)
            } else {
                val lowRank = linear(lowRankDim, dim, false, vb.pp("low_rank"))
                val embeddings = MaybeQuantizedEmbedding.new(inVocabSize, lowRankDim, vb)
                LowRankEmbeddings(embeddings, lowRank)
            }
    }

    override fun forward(xs: Tensor): Tensor {
        val embs = xs.apply(embeddings)
        return if (lowRank == null) embs else embs.apply(lowRank)
    }
}

internal class DepFormerSlice(
    val transformer: StreamingTransformer,
    val emb: LowRankEmbeddings,
    val linearIn: MaybeQuantizedLinear,
    val linearOut: MaybeQuantizedLinear,
) {
    companion object {
        fun new(
            inVocabSize: Int,
            outVocabSize: Int,
            mainTransformerDim: Int,
            cfg: DepFormerConfig,
            vb: MaybeQuantizedVarBuilder,
        ): DepFormerSlice {
            val dim = cfg.transformer.dModel
            val transformer = StreamingTransformer.new(cfg.transformer, vb.pp("transformer"))
            val emb = LowRankEmbeddings.new(inVocabSize, dim, cfg.lowRankEmbeddings, vb.pp("emb"))
            val linearIn = linear(mainTransformerDim, dim, false, vb.pp("linear_in"))
            val linearOut = linear(dim, outVocabSize, false, vb.pp("linear_out"))
            return DepFormerSlice(transformer, emb, linearIn, linearOut)
        }
    }
}

public class DepFormer private constructor(
    internal val slices: List<DepFormerSlice>,
) {
    public companion object {
        public fun new(
            textVocabSize: Int,
            audioVocabSize: Int,
            mainTransformerDim: Int,
            cfg: DepFormerConfig,
            vb: MaybeQuantizedVarBuilder,
        ): DepFormer {
            val slices = ArrayList<DepFormerSlice>(cfg.numSlices)
            for (sliceIdx in 0 until cfg.numSlices) {
                val inVs = if (sliceIdx == 0) textVocabSize else audioVocabSize
                val slice =
                    DepFormerSlice.new(
                        inVocabSize = inVs,
                        outVocabSize = audioVocabSize - 1,
                        mainTransformerDim = mainTransformerDim,
                        cfg = cfg,
                        vb = vb.pp(sliceIdx),
                    )
                slices.add(slice)
            }
            return DepFormer(slices)
        }
    }

    /**
     * Run a transformer sampling step, getting a token id per codebook.
     * - `xs` is the previous layer hidden state.
     */
    public fun sample(
        xs: Tensor,
        textToken: UInt?,
        forcedAudioTokens: List<UInt?>,
        lp: LogitsProcessor,
    ): UIntArray {
        val dev = xs.device()
        val tokens = UIntArray(slices.size)
        var lastToken = textToken
        for (sliceIdx in slices.indices) {
            if (sliceIdx == 0) {
                slices[sliceIdx].transformer.resetState()
            } else {
                slices[sliceIdx].transformer.copyState(slices[sliceIdx - 1].transformer)
            }
            val slice = slices[sliceIdx]
            var curXs = slice.linearIn.forward(xs)
            val lt = lastToken
            if (lt != null) {
                val tokenId = Tensor.fromSlice(uintArrayOf(lt), Shape(1, 1), dev)
                val tokenEmb = slice.emb.forward(tokenId)
                curXs = curXs.broadcastAdd(tokenEmb)
            }
            curXs = slice.transformer.forward(curXs)
            val rawLogits = curXs.apply(slice.linearOut)
            val logits =
                when (val bSize = rawLogits.dim(0)) {
                    1 -> rawLogits.i(0).i(0)
                    else -> bail("unexpected batch size $bSize")
                }
            val token = lp.sample(logits)
            tokens[sliceIdx] = token
            val forced = if (sliceIdx < forcedAudioTokens.size) forcedAudioTokens[sliceIdx] else null
            lastToken = forced ?: token
        }
        return tokens
    }

    public fun sampleCfg(
        xs: Tensor,
        cfgAlpha: Double,
        textToken: UInt?,
        forcedAudioTokens: List<UInt?>,
        lp: LogitsProcessor,
    ): UIntArray {
        val dev = xs.device()
        val tokens = UIntArray(slices.size)
        var lastToken = textToken
        for (sliceIdx in slices.indices) {
            if (sliceIdx == 0) {
                slices[sliceIdx].transformer.resetState()
            } else {
                slices[sliceIdx].transformer.copyState(slices[sliceIdx - 1].transformer)
            }
            val slice = slices[sliceIdx]
            var curXs = slice.linearIn.forward(xs)
            val lt = lastToken
            if (lt != null) {
                val tokenId = Tensor.fromSlice(uintArrayOf(lt), Shape(1, 1), dev)
                val tokenEmb = slice.emb.forward(tokenId)
                curXs = curXs.broadcastAdd(tokenEmb)
            }
            curXs = slice.transformer.forward(curXs)
            val rawLogits = curXs.apply(slice.linearOut)
            val logits =
                when (val bSize = rawLogits.dim(0)) {
                    2 -> {
                        val l0 = rawLogits.i(0).i(0).affine(cfgAlpha, 0.0)
                        val l1 = rawLogits.i(1).i(0).affine(cfgAlpha - 1.0, 0.0)
                        l0.sub(l1)
                    }

                    else -> {
                        bail("unexpected batch size $bSize")
                    }
                }
            val token = lp.sample(logits)
            tokens[sliceIdx] = token
            val forced = if (sliceIdx < forcedAudioTokens.size) forcedAudioTokens[sliceIdx] else null
            lastToken = forced ?: token
        }
        return tokens
    }
}

private sealed class LmStreamingTransformer : StreamingModule {
    data class Normal(
        val transformer: StreamingTransformer,
    ) : LmStreamingTransformer()

    data class Batched(
        val transformer: BatchedStreamingTransformer,
    ) : LmStreamingTransformer()

    override fun resetState() {
        when (this) {
            is Normal -> transformer.resetState()
            is Batched -> transformer.resetState()
        }
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor =
        when (this) {
            is Normal -> transformer.step(xs, mask)
            is Batched -> transformer.step(xs, mask)
        }

    fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        when (this) {
            is Normal -> transformer.resetBatchIdx(batchIdx, batchSize)
            is Batched -> transformer.resetBatchIdx(batchIdx)
        }
    }

    fun maybePrecomputeCaKv(caSrc: CaSrc?): CaSrc? =
        when (this) {
            is Normal -> transformer.maybePrecomputeCaKv(caSrc)
            is Batched -> transformer.maybePrecomputeCaKv(caSrc)
        }

    fun forward(
        xs: Tensor,
        m: StreamMask,
    ): Tensor =
        when (this) {
            is Normal -> transformer.forward(xs)
            is Batched -> transformer.forward(xs, m)
        }

    fun forwardCa(
        xs: Tensor,
        caSrc: CaSrc?,
        m: StreamMask,
    ): Tensor =
        when (this) {
            is Normal -> transformer.forwardCa(xs, caSrc)
            is Batched -> transformer.forwardCa(xs, caSrc, m)
        }
}

public class LmModel private constructor(
    private val transformer: LmStreamingTransformer,
    private val textEmb: MaybeQuantizedEmbedding,
    private val audioEmbs: List<MaybeQuantizedEmbedding>,
    private val textLinear: MaybeQuantizedLinear,
    private val outNorm: TransformerNorm,
    private val depformer: DepFormer?,
    private val audioVocabSize: Int,
    private val textInVocabSize: Int,
    private val conditionProvider: ConditionProvider?,
    private val extraHeadsList: List<MaybeQuantizedLinear>,
    private val dtype: DType,
) {
    public companion object {
        public fun new(
            cfg: LmConfig,
            vb: MaybeQuantizedVarBuilder,
        ): LmModel = newInternal(null, cfg, vb)

        public fun batched(
            batchSize: Int,
            cfg: LmConfig,
            vb: MaybeQuantizedVarBuilder,
        ): LmModel = newInternal(batchSize, cfg, vb)

        public fun newInternal(
            batchSize: Int?,
            cfg: LmConfig,
            vb: MaybeQuantizedVarBuilder,
        ): LmModel {
            val dModel = cfg.transformer.dModel
            val depformer =
                cfg.depformer?.let { dfCfg ->
                    DepFormer.new(
                        textVocabSize = cfg.textInVocabSize,
                        audioVocabSize = cfg.audioVocabSize,
                        mainTransformerDim = dModel,
                        cfg = dfCfg,
                        vb = vb.pp("depformer"),
                    )
                }
            val textEmb = MaybeQuantizedEmbedding.new(cfg.textInVocabSize, dModel, vb.pp("text_emb"))
            val outNorm = TransformerNorm.new(dModel, cfg.transformer, vb.pp("out_norm"))
            val textLinear = linear(dModel, cfg.textOutVocabSize, false, vb.pp("text_linear"))
            val transformer =
                if (batchSize == null) {
                    LmStreamingTransformer.Normal(
                        StreamingTransformer.new(cfg.transformer, vb.pp("transformer")),
                    )
                } else {
                    LmStreamingTransformer.Batched(
                        BatchedStreamingTransformer.new(batchSize, cfg.transformer, vb.pp("transformer")),
                    )
                }
            val vbE = vb.pp("emb")
            val audioEmbs =
                List(cfg.audioCodebooks) { i ->
                    MaybeQuantizedEmbedding.new(cfg.audioVocabSize, dModel, vbE.pp(i))
                }
            val dtype = vb.dtype()
            val conditionProvider =
                cfg.conditioners?.let { condCfg ->
                    ConditionProvider.new(dModel, condCfg, vb.pp("condition_provider"))
                }
            val extraHeadsList = ArrayList<MaybeQuantizedLinear>()
            if (cfg.extraHeads != null) {
                for (i in 0 until cfg.extraHeads.numHeads) {
                    val extraHead = linear(dModel, cfg.extraHeads.dim, false, vb.pp("extra_heads").pp(i))
                    extraHeadsList.add(extraHead)
                }
            }
            return LmModel(
                transformer = transformer,
                textEmb = textEmb,
                audioEmbs = audioEmbs,
                textLinear = textLinear,
                outNorm = outNorm,
                depformer = depformer,
                audioVocabSize = cfg.audioVocabSize,
                textInVocabSize = cfg.textInVocabSize,
                conditionProvider = conditionProvider,
                extraHeadsList = extraHeadsList,
                dtype = dtype,
            )
        }
    }

    public fun conditionProvider(): ConditionProvider? = conditionProvider

    public fun resetState() {
        transformer.resetState()
    }

    public fun inAudioCodebooks(): Int = audioEmbs.size

    public fun audioPadToken(): UInt = (audioVocabSize - 1).toUInt()

    public fun textStartToken(): UInt = (textInVocabSize - 1).toUInt()

    public fun generatedAudioCodebooks(): Int = depformer?.slices?.size ?: 0

    public fun isQuantized(): Boolean =
        when (textLinear) {
            is MaybeQuantizedLinear.Quantized -> true
            is MaybeQuantizedLinear.Real -> false
        }

    public fun device(): Device = textEmb.embeddings().device()

    public fun dtype(): DType = textEmb.embeddings().dtype()

    public fun forward(
        textIds: Tensor?,
        audioIds: List<Tensor?>,
        mask: StreamMask,
    ): LmForwardOutput = forwardCond(textIds, audioIds, null, mask)

    public fun extraHeads(vs: Tensor): List<Tensor> =
        extraHeadsList.map { extraHead -> vs.apply(extraHead) }

    public fun forwardCond(
        textIds: Tensor?,
        audioIds: List<Tensor?>,
        conditions: Condition?,
        mask: StreamMask,
    ): LmForwardOutput {
        var emb =
            if (textIds != null) {
                textIds.apply(textEmb)
            } else {
                val dev = textEmb.embeddings().device()
                Tensor.zeros(Shape(1, 1, textEmb.hiddenSize()), dtype, dev)
            }
        val limit = minOf(audioEmbs.size, audioIds.size)
        for (i in 0 until limit) {
            val aIds = audioIds[i]
            if (aIds != null) {
                val e = aIds.apply(audioEmbs[i])
                emb = emb.add(e)
            }
        }
        if (conditions != null) {
            when (conditions) {
                is Condition.AddToInput -> emb = emb.broadcastAdd(conditions.tensor)
            }
        }
        val ys = transformer.forward(emb, mask).apply(outNorm)
        val logits = ys.apply(textLinear)
        return LmForwardOutput(logits, ys)
    }

    public fun maybePrecomputeCaKv(caSrc: CaSrc?): CaSrc? =
        if (caSrc == null) null else transformer.maybePrecomputeCaKv(caSrc)

    public fun forwardCa(
        textIds: Tensor?,
        audioIds: List<Tensor?>,
        caSrc: CaSrc,
        conditions: Condition?,
        mask: StreamMask,
    ): LmForwardOutput {
        val bSize =
            when (caSrc) {
                is CaSrc.KeysValues -> caSrc.k.dim(0)
                is CaSrc.Tokens -> caSrc.tokens.dim(0)
            }
        var emb =
            if (textIds != null) {
                textIds.apply(textEmb)
            } else {
                val dev = textEmb.embeddings().device()
                Tensor.zeros(Shape(bSize, 1, textEmb.hiddenSize()), dtype, dev)
            }
        val limit = minOf(audioEmbs.size, audioIds.size)
        for (i in 0 until limit) {
            val aIds = audioIds[i]
            if (aIds != null) {
                val e = aIds.apply(audioEmbs[i])
                emb = emb.broadcastAdd(e)
            }
        }
        if (conditions != null) {
            when (conditions) {
                is Condition.AddToInput -> emb = emb.broadcastAdd(conditions.tensor)
            }
        }
        val ys = transformer.forwardCa(emb, caSrc, mask).apply(outNorm)
        val logits = ys.apply(textLinear)
        return LmForwardOutput(logits, ys)
    }

    public fun depformerSample(
        xs: Tensor,
        textToken: UInt?,
        forcedAudioTokens: List<UInt?>,
        lp: LogitsProcessor,
    ): UIntArray? = depformer?.sample(xs, textToken, forcedAudioTokens, lp)

    public fun depformerSampleCfg(
        xs: Tensor,
        cfgAlpha: Double,
        textToken: UInt?,
        forcedAudioTokens: List<UInt?>,
        lp: LogitsProcessor,
    ): UIntArray? = depformer?.sampleCfg(xs, cfgAlpha, textToken, forcedAudioTokens, lp)

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        transformer.resetBatchIdx(batchIdx, batchSize)
    }
}

public fun loadLmModel(
    cfg: LmConfig,
    modelFile: String,
    dtype: DType,
    dev: Device,
): LmModel {
    val quantized = modelFile.endsWith(".gguf")
    val vb =
        if (quantized) {
            MaybeQuantizedVarBuilder.Quantized(QuantizedVarBuilder.fromGguf(modelFile, dev))
        } else {
            MaybeQuantizedVarBuilder.Real(
                VarBuilder.fromMmapedSafetensors(listOf(modelFile), dtype, dev),
            )
        }
    return LmModel.new(cfg, vb)
}

public fun loadLm(
    modelFile: String,
    dtype: DType,
    dev: Device,
): LmModel = loadLmModel(LmConfig.v01(), modelFile, dtype, dev)

public fun loadLmStreaming(
    modelFile: String,
    dtype: DType,
    dev: Device,
): LmModel = loadLmModel(LmConfig.v01Streaming(8), modelFile, dtype, dev)

public fun loadLmStreamingBothWays(
    modelFile: String,
    dtype: DType,
    dev: Device,
): LmModel = loadLmModel(LmConfig.v01Streaming(16), modelFile, dtype, dev)

public fun loadLmVision(
    modelFile: String,
    overrideCrossAttentionGating: CrossAttentionGating?,
    overrideCrossAttentionInDim: Int?,
    dtype: DType,
    dev: Device,
): LmModel {
    val baseCfg = LmConfig.v01VisionStreaming(8)
    val updatedCa =
        overrideCrossAttentionGating?.let { gating ->
            CrossAttentionConfig(
                gating = gating,
                norm = baseCfg.transformer.norm,
                kvInDim = overrideCrossAttentionInDim,
            )
        }
    val cfg = baseCfg.copy(transformer = baseCfg.transformer.copy(crossAttention = updatedCa))
    return loadLmModel(cfg, modelFile, dtype, dev)
}

public fun loadLmS2s(
    modelFile: String,
    dtype: DType,
    dev: Device,
): LmModel = loadLmModel(LmConfig.s2s2b16rvq202501(), modelFile, dtype, dev)

public fun loadLmAsr(
    modelFile: String,
    dtype: DType,
    dev: Device,
): LmModel = loadLmModel(LmConfig.asrV011b(), modelFile, dtype, dev)

public class ForcedAudioTokens(
    private val acousticDelay: Int,
    private val preDelayTokens: List<UInt?>,
) {
    public companion object {
        public fun new(
            acousticDelay: Int,
            audioPadToken: UInt,
            streamCodebooks: IntArray,
        ): ForcedAudioTokens {
            val preDelayTokens = ArrayList<UInt?>()
            for (codebooks in streamCodebooks) {
                for (c in 0 until codebooks) {
                    val token = if (c == 0) null else audioPadToken
                    preDelayTokens.add(token)
                }
            }
            return ForcedAudioTokens(acousticDelay, preDelayTokens)
        }
    }

    public fun forcedTokens(stepIdx: Int): List<UInt?> =
        if (stepIdx < acousticDelay) {
            preDelayTokens
        } else {
            emptyList()
        }
}
