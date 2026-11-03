// port-lint: source transformer.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candle.i
import io.github.kotlinmania.candlenn.Activation
import io.github.kotlinmania.candlenn.Linear
import io.github.kotlinmania.candlenn.NonLinearity
import io.github.kotlinmania.candlenn.NonLinearityKind
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candlenn.rmsNorm
import io.github.kotlinmania.candlenn.ropeI
import io.github.kotlinmania.candlenn.sigmoid
import io.github.kotlinmania.candlenn.softmaxLastDim
import io.github.kotlinmania.serde.SerdeError
import io.github.kotlinmania.serde.SerdeException
import io.github.kotlinmania.serde.SerdeResult
import io.github.kotlinmania.serde.serdeCatching
import io.github.kotlinmania.serdecore.de.BooleanDeserialize
import io.github.kotlinmania.serdecore.de.Deserialize
import io.github.kotlinmania.serdecore.de.Deserializer
import io.github.kotlinmania.serdecore.de.EnumAccess
import io.github.kotlinmania.serdecore.de.F64Deserialize
import io.github.kotlinmania.serdecore.de.I32Deserialize
import io.github.kotlinmania.serdecore.de.MapAccess
import io.github.kotlinmania.serdecore.de.SeqAccess
import io.github.kotlinmania.serdecore.de.StringDeserialize
import io.github.kotlinmania.serdecore.de.Visitor
import io.github.kotlinmania.serdecore.ser.Serialize
import io.github.kotlinmania.serdecore.ser.Serializer
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

private val CROSS_ATTENTION_CONFIG_FIELDS: List<String> = listOf("gating", "norm", "kv_in_dim")
private val TRANSFORMER_CONFIG_FIELDS: List<String> =
    listOf(
        "d_model",
        "num_heads",
        "num_layers",
        "causal",
        "norm_first",
        "bias_ff",
        "bias_attn",
        "layer_scale",
        "positional_embedding",
        "use_conv_block",
        "cross_attention",
        "conv_kernel_size",
        "use_conv_bias",
        "gating",
        "norm",
        "context",
        "max_period",
        "max_seq_len",
        "kv_repeat",
        "dim_feedforward",
        "conv_layout",
        "shared_cross_attn",
    )
private val POSITIONAL_EMBEDDING_VARIANTS: List<String> = listOf("Rope", "Sin", "None")
private val CROSS_ATTENTION_GATING_VARIANTS: List<String> =
    listOf(
        "Normal",
        "ConstantGatedTanh",
        "ConstantGatedSigmoid",
        "ConditionalGatedTanh",
        "ConditionalGatedSigmoid",
        "ConditionalGatedSigmoidLearnableBias",
        "ConditionalGatedTanhLearnableBias",
    )

public data class CrossAttentionConfig(
    public val gating: CrossAttentionGating,
    public val norm: NormType,
    public val kvInDim: Int? = null,
) {
    public companion object : Deserialize<CrossAttentionConfig> {
        override fun <D> deserialize(deserializer: D): SerdeResult<CrossAttentionConfig>
            where D : Deserializer =
            deserializer.deserializeTuple(
                3,
                object : Visitor<CrossAttentionConfig> {
                    override fun expecting(): String = "tuple (CrossAttentionGating, NormType, Option<usize>)"

                    override fun <A> visitSeq(access: A): SerdeResult<CrossAttentionConfig>
                        where A : SeqAccess =
                        serdeCatching {
                            val optIntDe = I32Deserialize.asNullableDeserialize()
                            val gating =
                                access.nextElementSeed(SeedOf(CrossAttentionGating)).getOrThrow()
                                    ?: throw SerdeException(SerdeError.invalidLength(0, this))
                            val norm =
                                access.nextElementSeed(SeedOf(NormType)).getOrThrow()
                                    ?: throw SerdeException(SerdeError.invalidLength(1, this))
                            val kvInDim = access.nextElementSeed(SeedOf(optIntDe)).getOrThrow()
                            CrossAttentionConfig(gating = gating, norm = norm, kvInDim = kvInDim)
                        }

                    override fun <A> visitMap(access: A): SerdeResult<CrossAttentionConfig>
                        where A : MapAccess =
                        serdeCatching {
                            var gating: CrossAttentionGating? = null
                            var norm: NormType? = null
                            var kvInDim: Int? = null
                            val optIntDe = I32Deserialize.asNullableDeserialize()
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "gating" -> gating = access.nextValueSeed(SeedOf(CrossAttentionGating)).getOrThrow()
                                    "norm" -> norm = access.nextValueSeed(SeedOf(NormType)).getOrThrow()
                                    "kv_in_dim" -> kvInDim = access.nextValueSeed(SeedOf(optIntDe)).getOrThrow()
                                    else -> throw SerdeException(SerdeError.unknownField(key, CROSS_ATTENTION_CONFIG_FIELDS))
                                }
                            }
                            CrossAttentionConfig(
                                gating = gating ?: throw SerdeException(SerdeError.missingField("gating")),
                                norm = norm ?: throw SerdeException(SerdeError.missingField("norm")),
                                kvInDim = kvInDim,
                            )
                        }
                },
            )
    }
}

public data class TransformerConfig(
    public val dModel: Int,
    public val numHeads: Int,
    public val numLayers: Int,
    public val causal: Boolean,
    public val normFirst: Boolean,
    public val biasFf: Boolean,
    public val biasAttn: Boolean,
    public val layerScale: Double?,
    public val positionalEmbedding: PositionalEmbedding,
    public val useConvBlock: Boolean,
    public val crossAttention: CrossAttentionConfig?,
    public val convKernelSize: Int,
    public val useConvBias: Boolean,
    public val gating: Activation?,
    public val norm: NormType,
    public val context: Int,
    public val maxPeriod: Int,
    public val maxSeqLen: Int,
    public val kvRepeat: Int,
    public val dimFeedforward: Int,
    public val convLayout: Boolean,
    public val sharedCrossAttn: Boolean = false,
) {
    public companion object : Deserialize<TransformerConfig> {
        override fun <D> deserialize(deserializer: D): SerdeResult<TransformerConfig>
            where D : Deserializer =
            deserializer.deserializeStruct(
                "Config",
                TRANSFORMER_CONFIG_FIELDS,
                object : Visitor<TransformerConfig> {
                    override fun expecting(): String = "struct transformer::Config"

                    override fun <A> visitMap(access: A): SerdeResult<TransformerConfig>
                        where A : MapAccess =
                        serdeCatching {
                            var dModel: Int? = null
                            var numHeads: Int? = null
                            var numLayers: Int? = null
                            var causal: Boolean? = null
                            var normFirst: Boolean? = null
                            var biasFf: Boolean? = null
                            var biasAttn: Boolean? = null
                            var layerScale: Double? = null
                            var positionalEmbedding: PositionalEmbedding? = null
                            var useConvBlock: Boolean? = null
                            var crossAttention: CrossAttentionConfig? = null
                            var convKernelSize: Int? = null
                            var useConvBias: Boolean? = null
                            var gating: Activation? = null
                            var norm: NormType? = null
                            var context: Int? = null
                            var maxPeriod: Int? = null
                            var maxSeqLen: Int? = null
                            var kvRepeat: Int? = null
                            var dimFeedforward: Int? = null
                            var convLayout: Boolean? = null
                            var sharedCrossAttn = false
                            val optF64De = F64Deserialize.asNullableDeserialize()
                            val optCrossDe = CrossAttentionConfig.asNullableDeserialize()
                            val optActDe = Activation.asNullableDeserialize()
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "d_model" -> dModel = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "num_heads" -> numHeads = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "num_layers" -> numLayers = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "causal" -> causal = access.nextValueSeed(SeedOf(BooleanDeserialize)).getOrThrow()
                                    "norm_first" -> normFirst = access.nextValueSeed(SeedOf(BooleanDeserialize)).getOrThrow()
                                    "bias_ff" -> biasFf = access.nextValueSeed(SeedOf(BooleanDeserialize)).getOrThrow()
                                    "bias_attn" -> biasAttn = access.nextValueSeed(SeedOf(BooleanDeserialize)).getOrThrow()
                                    "layer_scale" -> layerScale = access.nextValueSeed(SeedOf(optF64De)).getOrThrow()
                                    "positional_embedding" -> positionalEmbedding = access.nextValueSeed(SeedOf(PositionalEmbedding)).getOrThrow()
                                    "use_conv_block" -> useConvBlock = access.nextValueSeed(SeedOf(BooleanDeserialize)).getOrThrow()
                                    "cross_attention" -> crossAttention = access.nextValueSeed(SeedOf(optCrossDe)).getOrThrow()
                                    "conv_kernel_size" -> convKernelSize = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "use_conv_bias" -> useConvBias = access.nextValueSeed(SeedOf(BooleanDeserialize)).getOrThrow()
                                    "gating" -> gating = access.nextValueSeed(SeedOf(optActDe)).getOrThrow()
                                    "norm" -> norm = access.nextValueSeed(SeedOf(NormType)).getOrThrow()
                                    "context" -> context = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "max_period" -> maxPeriod = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "max_seq_len" -> maxSeqLen = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "kv_repeat" -> kvRepeat = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "dim_feedforward" -> dimFeedforward = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "conv_layout" -> convLayout = access.nextValueSeed(SeedOf(BooleanDeserialize)).getOrThrow()
                                    "shared_cross_attn" -> sharedCrossAttn = access.nextValueSeed(SeedOf(BooleanDeserialize)).getOrThrow()
                                    else -> throw SerdeException(SerdeError.unknownField(key, TRANSFORMER_CONFIG_FIELDS))
                                }
                            }
                            TransformerConfig(
                                dModel = dModel ?: throw SerdeException(SerdeError.missingField("d_model")),
                                numHeads = numHeads ?: throw SerdeException(SerdeError.missingField("num_heads")),
                                numLayers = numLayers ?: throw SerdeException(SerdeError.missingField("num_layers")),
                                causal = causal ?: throw SerdeException(SerdeError.missingField("causal")),
                                normFirst = normFirst ?: throw SerdeException(SerdeError.missingField("norm_first")),
                                biasFf = biasFf ?: throw SerdeException(SerdeError.missingField("bias_ff")),
                                biasAttn = biasAttn ?: throw SerdeException(SerdeError.missingField("bias_attn")),
                                layerScale = layerScale,
                                positionalEmbedding =
                                    positionalEmbedding ?: throw SerdeException(SerdeError.missingField("positional_embedding")),
                                useConvBlock = useConvBlock ?: throw SerdeException(SerdeError.missingField("use_conv_block")),
                                crossAttention = crossAttention,
                                convKernelSize = convKernelSize ?: throw SerdeException(SerdeError.missingField("conv_kernel_size")),
                                useConvBias = useConvBias ?: throw SerdeException(SerdeError.missingField("use_conv_bias")),
                                gating = gating,
                                norm = norm ?: throw SerdeException(SerdeError.missingField("norm")),
                                context = context ?: throw SerdeException(SerdeError.missingField("context")),
                                maxPeriod = maxPeriod ?: throw SerdeException(SerdeError.missingField("max_period")),
                                maxSeqLen = maxSeqLen ?: throw SerdeException(SerdeError.missingField("max_seq_len")),
                                kvRepeat = kvRepeat ?: throw SerdeException(SerdeError.missingField("kv_repeat")),
                                dimFeedforward = dimFeedforward ?: throw SerdeException(SerdeError.missingField("dim_feedforward")),
                                convLayout = convLayout ?: throw SerdeException(SerdeError.missingField("conv_layout")),
                                sharedCrossAttn = sharedCrossAttn,
                            )
                        }
                },
            )
    }
}

public enum class PositionalEmbedding : Serialize {
    Rope,
    Sin,
    None,
    ;

    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
        when (this) {
            Rope -> serializer.serializeUnitVariant("PositionalEmbedding", 0u, "Rope")
            Sin -> serializer.serializeUnitVariant("PositionalEmbedding", 1u, "Sin")
            None -> serializer.serializeUnitVariant("PositionalEmbedding", 2u, "None")
        }

    public companion object : Deserialize<PositionalEmbedding> {
        override fun <D> deserialize(deserializer: D): SerdeResult<PositionalEmbedding>
            where D : Deserializer =
            deserializer.deserializeEnum(
                "PositionalEmbedding",
                POSITIONAL_EMBEDDING_VARIANTS,
                object : Visitor<PositionalEmbedding> {
                    override fun expecting(): String = "enum PositionalEmbedding"

                    override fun visitStr(v: String): SerdeResult<PositionalEmbedding> =
                        when (v) {
                            "Rope" -> SerdeResult.success(Rope)
                            "Sin" -> SerdeResult.success(Sin)
                            "None" -> SerdeResult.success(None)
                            else -> SerdeResult.failure(SerdeError.unknownVariant(v, POSITIONAL_EMBEDDING_VARIANTS))
                        }

                    override fun visitString(v: String): SerdeResult<PositionalEmbedding> = visitStr(v)

                    override fun <A> visitEnum(access: A): SerdeResult<PositionalEmbedding>
                        where A : EnumAccess =
                        serdeCatching {
                            val (variant, variantAccess) = access.variantSeed(SeedOf(StringDeserialize)).getOrThrow()
                            variantAccess.unitVariant().getOrThrow()
                            visitStr(variant).getOrThrow()
                        }
                },
            )
    }
}

public enum class CrossAttentionGating : Serialize {
    Normal,
    ConstantGatedTanh,
    ConstantGatedSigmoid,
    ConditionalGatedTanh,
    ConditionalGatedSigmoid,
    ConditionalGatedSigmoidLearnableBias,
    ConditionalGatedTanhLearnableBias,
    ;

    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
        serializer.serializeUnitVariant("CrossAttentionGating", ordinal.toUInt(), name)

    public companion object : Deserialize<CrossAttentionGating> {
        override fun <D> deserialize(deserializer: D): SerdeResult<CrossAttentionGating>
            where D : Deserializer =
            deserializer.deserializeEnum(
                "CrossAttentionGating",
                CROSS_ATTENTION_GATING_VARIANTS,
                object : Visitor<CrossAttentionGating> {
                    override fun expecting(): String = "enum CrossAttentionGating"

                    override fun visitStr(v: String): SerdeResult<CrossAttentionGating> =
                        entries.firstOrNull { it.name == v }?.let { SerdeResult.success(it) }
                            ?: SerdeResult.failure(SerdeError.unknownVariant(v, CROSS_ATTENTION_GATING_VARIANTS))

                    override fun visitString(v: String): SerdeResult<CrossAttentionGating> = visitStr(v)

                    override fun <A> visitEnum(access: A): SerdeResult<CrossAttentionGating>
                        where A : EnumAccess =
                        serdeCatching {
                            val (variant, variantAccess) = access.variantSeed(SeedOf(StringDeserialize)).getOrThrow()
                            variantAccess.unitVariant().getOrThrow()
                            visitStr(variant).getOrThrow()
                        }
                },
            )
    }
}

public sealed class CaSrc {
    public data class Tokens(
        public val tokens: Tensor,
    ) : CaSrc()

    public data class KeysValues(
        public val k: Tensor,
        public val v: Tensor,
    ) : CaSrc()
}

public class LayerScale(
    private val scale: Tensor,
) : Module {
    public companion object {
        public fun new(
            dModel: Int,
            init: Double,
            vb: MaybeQuantizedVarBuilder,
        ): LayerScale {
            val scale = vb.getUnquantized(dModel, "scale")
            return LayerScale(scale)
        }
    }

    override fun forward(xs: Tensor): Tensor = xs.broadcastMul(scale)
}

public sealed class XaGate : Module {
    public data object Normal : XaGate()

    public data class ConstantGated(
        public val alpha: Tensor,
    ) : XaGate()

    public data class ConditionalGated(
        public val inProj: MaybeQuantizedLinear,
        public val outProj: MaybeQuantizedLinear,
        public val activation: NonLinearity,
        public val learnableBias: Boolean,
    ) : XaGate()

    public companion object {
        public fun new(
            cfg: TransformerConfig,
            vb: MaybeQuantizedVarBuilder,
        ): XaGate {
            val gatingCfg =
                cfg.crossAttention?.gating
                    ?: bail("no cross-attention specified")
            return when (gatingCfg) {
                CrossAttentionGating.Normal -> {
                    Normal
                }

                CrossAttentionGating.ConstantGatedTanh -> {
                    val alpha = vb.getUnquantized(Shape(1, 1, 1), "alpha").tanh()
                    ConstantGated(alpha)
                }

                CrossAttentionGating.ConstantGatedSigmoid -> {
                    val alpha = sigmoid(vb.getUnquantized(Shape(1, 1, 1), "alpha").affine(1.0, -4.0))
                    ConstantGated(alpha)
                }

                CrossAttentionGating.ConditionalGatedTanh,
                CrossAttentionGating.ConditionalGatedSigmoid,
                CrossAttentionGating.ConditionalGatedSigmoidLearnableBias,
                CrossAttentionGating.ConditionalGatedTanhLearnableBias,
                -> {
                    val dim = cfg.dModel
                    val hiddenDims = floor(0.125f * dim.toFloat()).toInt()
                    val learnableBias =
                        gatingCfg == CrossAttentionGating.ConditionalGatedSigmoidLearnableBias ||
                            gatingCfg == CrossAttentionGating.ConditionalGatedTanhLearnableBias
                    val inProj = linear(dim, hiddenDims, false, vb.pp("alpha.0"))
                    val outProj = linear(hiddenDims, dim, learnableBias, vb.pp("alpha.2"))
                    val activation =
                        when (gatingCfg) {
                            CrossAttentionGating.ConditionalGatedTanh,
                            CrossAttentionGating.ConditionalGatedTanhLearnableBias,
                            -> NonLinearity.Tanh

                            CrossAttentionGating.ConditionalGatedSigmoid,
                            CrossAttentionGating.ConditionalGatedSigmoidLearnableBias,
                            -> NonLinearity.Sigmoid
                        }
                    ConditionalGated(
                        inProj = inProj,
                        outProj = outProj,
                        activation = activation,
                        learnableBias = learnableBias,
                    )
                }
            }
        }
    }

    override fun forward(xs: Tensor): Tensor =
        when (this) {
            is Normal -> {
                xs
            }

            is ConstantGated -> {
                xs.broadcastMul(alpha)
            }

            is ConditionalGated -> {
                val rawAlpha = xs.apply(inProj).relu().apply(outProj)
                val alpha =
                    when (activation.kind) {
                        NonLinearityKind.Tanh -> {
                            rawAlpha.tanh()
                        }

                        NonLinearityKind.Sigmoid -> {
                            if (learnableBias) {
                                sigmoid(rawAlpha)
                            } else {
                                sigmoid(rawAlpha.affine(1.0, -4.0))
                            }
                        }

                        else -> {
                            bail("Invalid non-linearity specified in cross-attention gating")
                        }
                    }
                xs.mul(alpha)
            }
        }
}

public class StreamingMultiheadCrossAttention(
    private val inProjQ: MaybeQuantizedLinear,
    private val inProjKv: MaybeQuantizedLinear,
    private val outProj: MaybeQuantizedLinear,
    private val kvRepeat: Int,
    private val numHeads: Int,
    private val gate: XaGate,
) {
    public companion object {
        public fun new(
            cfg: TransformerConfig,
            vb: MaybeQuantizedVarBuilder,
            gateVb: MaybeQuantizedVarBuilder? = null,
        ): StreamingMultiheadCrossAttention {
            val embedDim = cfg.dModel
            val numKv = cfg.numHeads / cfg.kvRepeat
            val outKvDim = numKv * (embedDim / cfg.numHeads)
            val outDim = embedDim + 2 * outKvDim
            val (inProjQ, inProjKv) =
                if (vb.containsKey("in_proj_weight")) {
                    when (vb) {
                        is MaybeQuantizedVarBuilder.Quantized -> {
                            bail("Quantized cross-attention layers require a separate in_proj_weight_q and in_proj_weight_kv")
                        }

                        is MaybeQuantizedVarBuilder.Real -> {
                            val weights = vb.weights
                            val inProjWeight = weights.get(Shape(outDim, embedDim), "in_proj_weight")
                            val inProjWeightQ = inProjWeight.narrow(0, 0, embedDim)
                            val inProjWeightKv = inProjWeight.narrow(0, embedDim, 2 * outKvDim)
                            val (inProjBiasQ, inProjBiasKv) =
                                if (cfg.biasAttn) {
                                    val b = weights.get(outDim, "in_proj_bias")
                                    b.narrow(0, 0, embedDim) to b.narrow(0, embedDim, 2 * outKvDim)
                                } else {
                                    null to null
                                }
                            MaybeQuantizedLinear.Real(Linear.new(inProjWeightQ, inProjBiasQ)) to
                                MaybeQuantizedLinear.Real(Linear.new(inProjWeightKv, inProjBiasKv))
                        }
                    }
                } else {
                    val caCfg =
                        cfg.crossAttention
                            ?: bail("cfg.cross_attention is None in cross_attention module")
                    val kvInDim =
                        when (val d = caCfg.kvInDim) {
                            null, 0 -> embedDim
                            else -> d
                        }
                    val inProjWeightQ = vb.get(Shape(embedDim, embedDim), "in_proj_weight_q")
                    val inProjWeightKv = vb.get(Shape(2 * outKvDim, kvInDim), "in_proj_weight_kv")
                    val (inProjBiasQ, inProjBiasKv) =
                        if (cfg.biasAttn) {
                            vb.getUnquantized(embedDim, "in_proj_bias_q") to
                                vb.getUnquantized(2 * outKvDim, "in_proj_bias_kv")
                        } else {
                            null to null
                        }
                    linearFrom(inProjWeightQ, inProjBiasQ) to linearFrom(inProjWeightKv, inProjBiasKv)
                }
            val outProj = linear(embedDim, embedDim, cfg.biasAttn, vb.pp("out_proj"))
            val gate =
                if (gateVb == null) {
                    XaGate.new(cfg, vb.pp("gate"))
                } else {
                    XaGate.new(cfg, gateVb)
                }
            return StreamingMultiheadCrossAttention(
                inProjQ = inProjQ,
                inProjKv = inProjKv,
                outProj = outProj,
                kvRepeat = cfg.kvRepeat,
                numHeads = cfg.numHeads,
                gate = gate,
            )
        }
    }

    public fun isQuantized(): Boolean =
        when (inProjQ) {
            is MaybeQuantizedLinear.Quantized -> true
            is MaybeQuantizedLinear.Real -> false
        }

    public fun computeKv(caSrc: CaSrc): KeyValuePair =
        when (caSrc) {
            is CaSrc.KeysValues -> {
                KeyValuePair(caSrc.k, caSrc.v)
            }

            is CaSrc.Tokens -> {
                val xs = caSrc.tokens
                val kv = xs.apply(inProjKv)
                val (caB, caT, caDim) = kv.dims3()
                val headDim = caDim / (2 * numHeads)
                val numKvHeads = caDim / (2 * headDim)
                val reshapedKv = kv.reshape(Shape(caB, caT, 2, numKvHeads, headDim))
                val castKv = if (isQuantized()) reshapedKv.toDtype(matmulDtype(xs.device())) else reshapedKv
                val k =
                    castKv
                        .narrow(2, 0, 1)
                        .squeeze(2)
                        .transpose(1, 2)
                        .contiguous()
                val v =
                    castKv
                        .narrow(2, 1, 1)
                        .squeeze(2)
                        .transpose(1, 2)
                        .contiguous()
                KeyValuePair(k, v)
            }
        }

    public fun forward(
        xs: Tensor,
        caSrc: CaSrc,
        mask: Tensor?,
    ): Tensor {
        if (kvRepeat != 1) {
            bail("only kv-repeat = 1 is supported")
        }
        val (b, t, hd) = xs.dims3()
        val headDim = hd / numHeads
        val rawQ = xs.apply(inProjQ)
        val originalDtype = rawQ.dtype()
        val reshapedQ = rawQ.reshape(Shape(b, t, numHeads, headDim))
        val castQ = if (isQuantized()) reshapedQ.toDtype(matmulDtype(xs.device())) else reshapedQ
        val kvPair = computeKv(caSrc)
        val q = castQ.transpose(1, 2).contiguous()
        var preWs = q.matmul(kvPair.k.t()).affine(headDim.toDouble().pow(-0.5), 0.0)
        if (mask != null) {
            preWs = preWs.broadcastAdd(mask)
        }
        val ws = softmaxLastDim(preWs)
        val attnOut = ws.matmul(kvPair.v)
        return attnOut
            .transpose(1, 2)
            .reshape(Shape(b, t, hd))
            .toDtype(originalDtype)
            .apply(outProj)
            .apply(gate)
    }
}

public class Rope(
    private val sin: Tensor,
    private val cos: Tensor,
) {
    public fun applyRotaryEmb(qk: Tensor): Tensor {
        val qkDtype = qk.dtype()
        return ropeI(qk.toDtype(DType.F32), cos, sin).toDtype(qkDtype)
    }
}

public class RotaryEmbedding(
    private val invFreq: Tensor,
) {
    public companion object {
        public fun new(
            dim: Int,
            theta: Float,
            dev: Device,
        ): RotaryEmbedding {
            val invFreqLen = (dim + 1) / 2
            val invFreq =
                FloatArray(invFreqLen) { idx ->
                    val i = idx * 2
                    1.0f / theta.pow(i.toFloat() / dim.toFloat())
                }
            val invFreqTensor = Tensor.fromSlice(invFreq, Shape(1, invFreqLen), dev)
            return RotaryEmbedding(invFreqTensor)
        }
    }

    public fun rope(pos: Tensor): Rope {
        val t = pos.toDtype(DType.F32)
        val dims = t.dims()
        val freqs =
            when (dims.size) {
                1 -> {
                    val d = dims[0]
                    t.reshape(Shape(d, 1)).matmul(invFreq)
                }

                2 -> {
                    val b = dims[0]
                    val d = dims[1]
                    val freqDim = invFreq.dim(1)
                    t.reshape(Shape(b * d, 1)).matmul(invFreq).reshape(Shape(b, d, freqDim))
                }

                else -> {
                    bail("Invalid shape for rotary embedding $pos")
                }
            }
        return Rope(sin = freqs.sin(), cos = freqs.cos())
    }
}

private fun flashAttn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmaxScale: Float,
    causal: Boolean,
): Tensor = bail("compile with '--features flash-attn'")

public class StreamingMultiheadAttention(
    private val inProj: MaybeQuantizedLinear,
    private val outProj: MaybeQuantizedLinear,
    private val kvRepeat: Int,
    private val numHeads: Int,
    private val context: Int,
    internal var kvCache: KvCache,
    private val useFlashAttn: Boolean,
) {
    public companion object {
        public fun new(
            cfg: TransformerConfig,
            vb: MaybeQuantizedVarBuilder,
        ): StreamingMultiheadAttention {
            val embedDim = cfg.dModel
            val numKv = cfg.numHeads / cfg.kvRepeat
            val outDim = embedDim + 2 * numKv * (embedDim / cfg.numHeads)
            val inProjWeight = vb.get(Shape(outDim, embedDim), "in_proj_weight")
            val inProjBias = if (cfg.biasAttn) vb.getUnquantized(outDim, "in_proj_bias") else null
            val inProj = linearFrom(inProjWeight, inProjBias)
            val outProj = linear(embedDim, embedDim, cfg.biasAttn, vb.pp("out_proj"))
            return StreamingMultiheadAttention(
                inProj = inProj,
                outProj = outProj,
                kvRepeat = cfg.kvRepeat,
                numHeads = cfg.numHeads,
                context = cfg.context,
                kvCache = KvCache.new(2, cfg.context),
                useFlashAttn = false,
            )
        }
    }

    public fun isQuantized(): Boolean =
        when (inProj) {
            is MaybeQuantizedLinear.Quantized -> true
            is MaybeQuantizedLinear.Real -> false
        }

    public fun forward(
        xs: Tensor,
        rope: Rope?,
        mask: Tensor?,
    ): Tensor {
        if (kvRepeat != 1) {
            bail("only kv-repeat = 1 is supported")
        }
        val (b, t, hd) = xs.dims3()
        val headDim = hd / numHeads
        val rawQkv = xs.apply(inProj).reshape(Shape(b, t, 3, numHeads, headDim))
        val originalDtype = rawQkv.dtype()
        val qkv = if (isQuantized()) rawQkv.toDtype(matmulDtype(xs.device())) else rawQkv
        var q =
            qkv
                .narrow(2, 0, 1)
                .squeeze(2)
                .transpose(1, 2)
                .contiguous()
        var k =
            qkv
                .narrow(2, 1, 1)
                .squeeze(2)
                .transpose(1, 2)
                .contiguous()
        val v =
            qkv
                .narrow(2, 2, 1)
                .squeeze(2)
                .transpose(1, 2)
                .contiguous()
        if (rope != null) {
            q = rope.applyRotaryEmb(q)
            k = rope.applyRotaryEmb(k)
        }
        val kvPair = kvCache.append(k.contiguous(), v.contiguous())
        val cachedK = kvPair.k
        val cachedV = kvPair.v
        val kLen = cachedK.dim(2)
        val kTargetLen = t + min(context, kLen - t)
        val (trimmedK, trimmedV) =
            if (kTargetLen < kLen) {
                cachedK.narrow(2, kLen - kTargetLen, kTargetLen) to
                    cachedV.narrow(2, kLen - kTargetLen, kTargetLen)
            } else {
                cachedK to cachedV
            }
        val attnOut =
            if (q.dtype() == DType.BF16 && useFlashAttn) {
                val qT = q.transpose(1, 2)
                val kT = trimmedK.transpose(1, 2)
                val vT = trimmedV.transpose(1, 2)
                val softmaxScale = 1.0f / sqrt(headDim.toFloat())
                flashAttn(qT, kT, vT, softmaxScale, mask != null).transpose(1, 2)
            } else {
                var preWs = q.matmul(trimmedK.t()).affine(headDim.toDouble().pow(-0.5), 0.0)
                if (mask != null) {
                    preWs = preWs.broadcastAdd(mask)
                }
                val ws = softmaxLastDim(preWs)
                ws.matmul(trimmedV)
            }
        return attnOut
            .transpose(1, 2)
            .reshape(Shape(b, t, hd))
            .toDtype(originalDtype)
            .apply(outProj)
    }

    public fun resetKvCache() {
        kvCache.reset()
    }

    public fun setKvCache(kvCache: KvCache) {
        this.kvCache = kvCache
    }
}

public sealed class Mlp : Module {
    public data class NoGating(
        public val linear1: MaybeQuantizedLinear,
        public val linear2: MaybeQuantizedLinear,
    ) : Mlp()

    public data class Gating(
        public val linearIn: MaybeQuantizedLinear,
        public val linearOut: MaybeQuantizedLinear,
        public val activation: Activation,
    ) : Mlp()

    public companion object {
        public fun new(
            cfg: TransformerConfig,
            vb: MaybeQuantizedVarBuilder,
        ): Mlp {
            val dModel = cfg.dModel
            val gating = cfg.gating
            return if (gating == null) {
                val linear1 = linear(dModel, cfg.dimFeedforward, cfg.biasFf, vb.pp("linear1"))
                val linear2 = linear(cfg.dimFeedforward, dModel, cfg.biasFf, vb.pp("linear2"))
                NoGating(linear1, linear2)
            } else {
                val vbGating = vb.pp("gating")
                val hidden =
                    if (cfg.dimFeedforward == 4 * dModel) {
                        11 * dModel / 4
                    } else {
                        2 * cfg.dimFeedforward / 3
                    }
                val linearIn = linear(dModel, 2 * hidden, cfg.biasFf, vbGating.pp("linear_in"))
                val linearOut = linear(hidden, dModel, cfg.biasFf, vbGating.pp("linear_out"))
                Gating(linearIn, linearOut, gating)
            }
        }
    }

    override fun forward(xs: Tensor): Tensor =
        when (this) {
            is NoGating -> {
                xs.apply(linear1).geluErf().apply(linear2)
            }

            is Gating -> {
                val projected = xs.apply(linearIn)
                val (b, t, totalDim) = projected.dims3()
                val hiddenDim = totalDim / 2
                val reshaped = projected.reshape(Shape(b, t, 2, hiddenDim))
                val x0 = reshaped.narrow(2, 0, 1).squeeze(2).apply(activation)
                val x1 = reshaped.narrow(2, 1, 1).squeeze(2)
                x0.mul(x1).apply(linearOut)
            }
        }
}

public class RmsNorm(
    internal val alpha: Tensor,
    internal val eps: Float,
) : Module {
    public companion object {
        public fun new(
            dModel: Int,
            eps: Float,
            vb: MaybeQuantizedVarBuilder,
        ): RmsNorm {
            val alpha = vb.getUnquantized(Shape(1, 1, dModel), "alpha").reshape(Shape(dModel))
            return RmsNorm(alpha, eps)
        }
    }

    override fun forward(xs: Tensor): Tensor = rmsNorm(xs, alpha, eps)
}

public class LayerNorm(
    private val inner: io.github.kotlinmania.candlenn.LayerNorm,
) : Module {
    public companion object {
        public fun new(
            dModel: Int,
            eps: Float,
            vb: MaybeQuantizedVarBuilder,
        ): LayerNorm {
            val bias = vb.getUnquantized(dModel, "bias")
            val alpha =
                if (vb.containsKey("alpha")) {
                    vb.getUnquantized(Shape(1, 1, dModel), "alpha").reshape(Shape(dModel))
                } else {
                    vb.getUnquantized(dModel, "weight").reshape(Shape(dModel))
                }
            val inner =
                io.github.kotlinmania.candlenn.LayerNorm
                    .new(alpha, bias, eps.toDouble())
            return LayerNorm(inner)
        }
    }

    override fun forward(xs: Tensor): Tensor = inner.forward(xs)
}

public sealed class TransformerNorm : Module {
    public data class LayerNormVariant(
        public val norm: LayerNorm,
    ) : TransformerNorm()

    public data class RmsNormVariant(
        public val norm: RmsNorm,
    ) : TransformerNorm()

    public companion object {
        public fun new(
            dModel: Int,
            cfg: TransformerConfig,
            vb: MaybeQuantizedVarBuilder,
        ): TransformerNorm = newShortcut(dModel, cfg.norm, vb)

        public fun newShortcut(
            dModel: Int,
            typ: NormType,
            vb: MaybeQuantizedVarBuilder,
        ): TransformerNorm =
            when (typ) {
                NormType.LayerNorm -> LayerNormVariant(LayerNorm.new(dModel, 1e-5f, vb))
                NormType.RmsNorm -> RmsNormVariant(RmsNorm.new(dModel, 1e-8f, vb))
            }
    }

    override fun forward(xs: Tensor): Tensor =
        when (this) {
            is LayerNormVariant -> norm.forward(xs)
            is RmsNormVariant -> norm.forward(xs)
        }
}

public class CrossAttentionBlock(
    public val norm: TransformerNorm,
    public val attn: StreamingMultiheadCrossAttention,
)

public class StreamingTransformerLayer(
    internal val selfAttn: StreamingMultiheadAttention,
    private val mlp: Mlp,
    private val norm1: TransformerNorm,
    private val norm2: TransformerNorm,
    private val layerScale1: LayerScale?,
    private val layerScale2: LayerScale?,
    internal val crossAttn: CrossAttentionBlock?,
    private val normFirst: Boolean,
) {
    public companion object {
        public fun new(
            cfg: TransformerConfig,
            vb: MaybeQuantizedVarBuilder,
            sharedCaVb: MaybeQuantizedVarBuilder? = null,
        ): StreamingTransformerLayer {
            if (cfg.useConvBlock) {
                bail("conv-block is not supported")
            }
            val dModel = cfg.dModel
            val mlp = Mlp.new(cfg, vb)
            val norm1 = TransformerNorm.new(dModel, cfg, vb.pp("norm1"))
            val norm2 = TransformerNorm.new(dModel, cfg, vb.pp("norm2"))
            val layerScale1 = cfg.layerScale?.let { ls -> LayerScale.new(dModel, ls, vb.pp("layer_scale_1")) }
            val layerScale2 = cfg.layerScale?.let { ls -> LayerScale.new(dModel, ls, vb.pp("layer_scale_2")) }
            val selfAttn = StreamingMultiheadAttention.new(cfg, vb.pp("self_attn"))
            val crossAttn =
                cfg.crossAttention?.let { caCfg ->
                    val normCross = TransformerNorm.newShortcut(dModel, caCfg.norm, vb.pp("norm_cross"))
                    val ca =
                        if (sharedCaVb == null) {
                            StreamingMultiheadCrossAttention.new(cfg, vb.pp("cross_attention"), null)
                        } else {
                            StreamingMultiheadCrossAttention.new(
                                cfg,
                                sharedCaVb.pp("cross_attention"),
                                vb.pp("cross_attention.gate"),
                            )
                        }
                    CrossAttentionBlock(normCross, ca)
                }
            return StreamingTransformerLayer(
                selfAttn = selfAttn,
                mlp = mlp,
                norm1 = norm1,
                norm2 = norm2,
                layerScale1 = layerScale1,
                layerScale2 = layerScale2,
                crossAttn = crossAttn,
                normFirst = cfg.normFirst,
            )
        }
    }

    public fun forward(
        xs: Tensor,
        rope: Rope?,
        caSrc: CaSrc?,
        mask: Tensor?,
    ): Tensor {
        if (!normFirst) {
            bail("only norm_first = true is supported")
        }
        val n1 = xs.apply(norm1)
        val attnOut = selfAttn.forward(n1, rope, mask)
        val scaledAttnOut = if (layerScale1 != null) attnOut.apply(layerScale1) else attnOut
        var cur = xs.add(scaledAttnOut)

        if (crossAttn != null && caSrc != null) {
            val residual = cur
            val normCrossOut = cur.apply(crossAttn.norm)
            cur = residual.add(crossAttn.attn.forward(normCrossOut, caSrc, null))
        }

        val mlpOut = cur.apply(norm2).apply(mlp)
        val scaledMlpOut = if (layerScale2 != null) mlpOut.apply(layerScale2) else mlpOut
        return cur.add(scaledMlpOut)
    }

    public fun resetKvCache() {
        selfAttn.resetKvCache()
    }

    public fun setKvCache(kvCache: KvCache) {
        selfAttn.setKvCache(kvCache)
    }
}

public class StreamingTransformer internal constructor(
    private val layers: List<StreamingTransformerLayer>,
    private val positionalEmbedding: PositionalEmbedding,
    private val maxPeriod: Int,
    private val causal: Boolean,
    private val numHeads: Int,
    private val context: Int,
    private var lastResetPos: MutableList<Int>,
    private val rope: RotaryEmbedding?,
) : StreamingModule {
    public companion object {
        public fun new(
            cfg: TransformerConfig,
            vb: MaybeQuantizedVarBuilder,
        ): StreamingTransformer {
            val vbL = vb.pp("layers")
            val rope =
                when (cfg.positionalEmbedding) {
                    PositionalEmbedding.Rope -> {
                        RotaryEmbedding.new(
                            dim = cfg.dModel / cfg.numHeads,
                            theta = cfg.maxPeriod.toFloat(),
                            dev = vb.device(),
                        )
                    }

                    PositionalEmbedding.None, PositionalEmbedding.Sin -> {
                        null
                    }
                }
            val layers = ArrayList<StreamingTransformerLayer>(cfg.numLayers)
            for (layerIdx in 0 until cfg.numLayers) {
                val sharedVb = if (cfg.sharedCrossAttn) vbL.pp(0) else null
                val layer = StreamingTransformerLayer.new(cfg, vbL.pp(layerIdx), sharedVb)
                layers.add(layer)
            }
            return StreamingTransformer(
                layers = layers,
                positionalEmbedding = cfg.positionalEmbedding,
                maxPeriod = cfg.maxPeriod,
                causal = cfg.causal,
                numHeads = cfg.numHeads,
                context = cfg.context,
                lastResetPos = mutableListOf(),
                rope = rope,
            )
        }
    }

    public fun forward(xs: Tensor): Tensor = forwardCa(xs, null)

    private fun currentSeqLen(): Int = layers[0].selfAttn.kvCache.currentSeqLen()

    public fun forwardCa(
        xs: Tensor,
        caSrc: CaSrc?,
    ): Tensor {
        val (b, t, c) = xs.dims3()
        if (!causal) {
            bail("only causal mode is supported")
        }
        if (lastResetPos.isEmpty()) {
            repeat(b) {
                lastResetPos.add(0)
            }
        }
        val curSeqLen = currentSeqLen()
        val ks = layers[0].selfAttn.kvCache.positions(t)
        val minKs = ks.minOrNull() ?: bail("no positions, is t == 0?")
        val mask =
            if (t == 1 && lastResetPos.all { v -> v <= minKs }) {
                null
            } else {
                val maskValues = FloatArray(b * t * ks.size)
                var idx = 0
                for (resetPos in lastResetPos) {
                    for (tIdx in 0 until t) {
                        val tPos = tIdx + curSeqLen
                        for (kPos in ks) {
                            maskValues[idx++] =
                                if (resetPos <= kPos && kPos <= tPos && tPos <= kPos + context) {
                                    0.0f
                                } else {
                                    Float.NEGATIVE_INFINITY
                                }
                        }
                    }
                }
                Tensor
                    .fromSlice(maskValues, Shape(b, 1, t, ks.size), xs.device())
                    .toDtype(xs.dtype())
                    .expand(Shape(b, numHeads, t, ks.size))
            }
        val pos = Tensor.arange(curSeqLen, curSeqLen + t, xs.device())
        val currentRope = rope?.rope(pos)
        var cur =
            when (positionalEmbedding) {
                PositionalEmbedding.Rope, PositionalEmbedding.None -> {
                    xs
                }

                PositionalEmbedding.Sin -> {
                    val dev = xs.device()
                    val theta = maxPeriod.toFloat()
                    val halfDim = c / 2
                    val positions = pos.unsqueeze(1).toDtype(DType.F32)
                    val invFreq =
                        FloatArray(halfDim) { i ->
                            1.0f / theta.pow(i.toFloat() / (halfDim - 1).toFloat())
                        }
                    val invFreqTensor = Tensor.fromSlice(invFreq, Shape(1, halfDim), dev)
                    val freqs = positions.broadcastMul(invFreqTensor)
                    val posEmb = Tensor.cat(listOf(freqs.cos(), freqs.sin()), D.Minus1)
                    xs.broadcastAdd(posEmb)
                }
            }
        for (layer in layers) {
            cur = layer.forward(cur, currentRope, caSrc, mask)
        }
        return cur
    }

    public fun maybePrecomputeCaKv(caSrc: CaSrc?): CaSrc? =
        when (caSrc) {
            null -> {
                null
            }

            is CaSrc.KeysValues -> {
                caSrc
            }

            is CaSrc.Tokens -> {
                if (layers.isEmpty()) {
                    caSrc
                } else {
                    val caBlock = layers[0].crossAttn
                    if (caBlock == null) {
                        caSrc
                    } else {
                        val kv = caBlock.attn.computeKv(caSrc)
                        CaSrc.KeysValues(kv.k, kv.v)
                    }
                }
            }
        }

    public fun copyState(from: StreamingTransformer) {
        if (layers.size != from.layers.size) {
            bail("cannot copy kv-caches as the transformers have different depths")
        }
        lastResetPos = from.lastResetPos.toMutableList()
        for (i in layers.indices) {
            layers[i].setKvCache(
                from.layers[i]
                    .selfAttn.kvCache
                    .copy(),
            )
        }
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        if (lastResetPos.isEmpty()) {
            repeat(batchSize) {
                lastResetPos.add(0)
            }
        }
        if (batchIdx >= lastResetPos.size) {
            bail("batch_idx $batchIdx is out of bounds for last_reset_pos")
        }
        lastResetPos[batchIdx] = currentSeqLen()
    }

    override fun resetState() {
        lastResetPos.clear()
        for (layer in layers) {
            layer.resetKvCache()
        }
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor {
        val t = xs.asOption() ?: return StreamTensor.empty()
        return StreamTensor.fromTensor(forward(t))
    }
}

public class ProjectedTransformer(
    private val transformer: StreamingTransformer,
    private val inputProj: MaybeQuantizedLinear?,
    private val outputProjs: List<MaybeQuantizedLinear?>,
    private val convLayout: Boolean,
) : StreamingModule {
    public companion object {
        public fun new(
            inputDim: Int,
            outputDims: IntArray,
            cfg: TransformerConfig,
            vb: MaybeQuantizedVarBuilder,
        ): ProjectedTransformer {
            val transformer = StreamingTransformer.new(cfg, vb.pp("transformer"))
            val inputProj =
                if (inputDim == cfg.dModel) {
                    null
                } else {
                    linear(inputDim, cfg.dModel, false, vb.pp("input_proj"))
                }
            val outputProjs = ArrayList<MaybeQuantizedLinear?>(outputDims.size)
            val vbO = vb.pp("output_projs")
            for ((i, outputDim) in outputDims.withIndex()) {
                val outputProj =
                    if (outputDim == cfg.dModel) {
                        null
                    } else {
                        linear(cfg.dModel, outputDim, false, vbO.pp(i))
                    }
                outputProjs.add(outputProj)
            }
            return ProjectedTransformer(
                transformer = transformer,
                inputProj = inputProj,
                outputProjs = outputProjs,
                convLayout = cfg.convLayout,
            )
        }
    }

    public fun forward(xs: Tensor): List<Tensor> {
        var cur = if (convLayout) xs.transpose(1, 2) else xs
        if (inputProj != null) {
            cur = cur.apply(inputProj)
        }
        cur = transformer.forward(cur)
        val ys = ArrayList<Tensor>(outputProjs.size)
        for (outputProj in outputProjs) {
            var y = if (outputProj != null) cur.apply(outputProj) else cur
            if (convLayout) {
                y = y.transpose(1, 2)
            }
            ys.add(y)
        }
        return ys
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        transformer.resetBatchIdx(batchIdx, batchSize)
    }

    override fun resetState() {
        transformer.resetState()
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor {
        val raw = xs.asOption() ?: return StreamTensor.empty()
        val transposedIn = if (convLayout) raw.transpose(1, 2) else raw
        val projectedIn = StreamTensor.fromTensor(transposedIn).apply(inputProj)
        val transformed = transformer.step(projectedIn, mask)
        val projectedOut = transformed.apply(outputProjs[0])
        val outOpt = projectedOut.asOption() ?: return StreamTensor.empty()
        val finalOut = if (convLayout) outOpt.transpose(1, 2) else outOpt
        return StreamTensor.fromTensor(finalOut)
    }
}

public sealed class Transformer : StreamingModule {
    public data class Standard(
        public val transformer: ProjectedTransformer,
    ) : Transformer()

    public data class Batched(
        public val transformer: BatchedProjectedTransformer,
    ) : Transformer()

    public companion object {
        public fun new(
            batchSize: Int?,
            dim: Int,
            cfg: TransformerConfig,
            vb: VarBuilder,
        ): Transformer =
            if (batchSize != null) {
                Batched(
                    BatchedProjectedTransformer.new(
                        inputDim = dim,
                        outputDims = intArrayOf(dim),
                        batchSize = batchSize,
                        cfg = cfg,
                        vb = MaybeQuantizedVarBuilder.Real(vb),
                    ),
                )
            } else {
                Standard(
                    ProjectedTransformer.new(
                        inputDim = dim,
                        outputDims = intArrayOf(dim),
                        cfg = cfg,
                        vb = MaybeQuantizedVarBuilder.Real(vb),
                    ),
                )
            }
    }

    public fun forward(xs: Tensor): List<Tensor> =
        when (this) {
            is Standard -> transformer.forward(xs)
            is Batched -> transformer.forward(xs, StreamMask.empty())
        }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        when (this) {
            is Standard -> transformer.resetBatchIdx(batchIdx, batchSize)
            is Batched -> transformer.resetBatchIdx(batchIdx)
        }
    }

    override fun resetState() {
        when (this) {
            is Standard -> transformer.resetState()
            is Batched -> transformer.resetState()
        }
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor =
        when (this) {
            is Standard -> transformer.step(xs, mask)
            is Batched -> transformer.step(xs, mask)
        }
}
