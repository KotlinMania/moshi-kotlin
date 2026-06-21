// port-lint: source moshi-core/src/transformer.rs
package io.github.kotlinmania.moshi

enum class PositionalEmbedding {
    ROPE,
    SIN,
    NONE,
}

enum class CrossAttentionGating {
    NORMAL,
    CONSTANT_GATED_TANH,
    CONSTANT_GATED_SIGMOID,
    CONDITIONAL_GATED_TANH,
    CONDITIONAL_GATED_SIGMOID,
    CONDITIONAL_GATED_SIGMOID_LEARNABLE_BIAS,
    CONDITIONAL_GATED_TANH_LEARNABLE_BIAS,
}

data class CrossAttentionGatingTriple(
    val gating: CrossAttentionGating,
    val normType: NormType,
    val inputDim: Int?,
)

data class TransformerConfig(
    val dModel: Int,
    val numHeads: Int,
    val numLayers: Int,
    val causal: Boolean,
    val normFirst: Boolean,
    val biasFf: Boolean,
    val biasAttn: Boolean,
    val layerScale: Double?,
    val positionalEmbedding: PositionalEmbedding,
    val useConvBlock: Boolean,
    val crossAttention: CrossAttentionGatingTriple?,
    val convKernelSize: Int,
    val useConvBias: Boolean,
    val gatingName: String?,
    val norm: NormType,
    val context: Int,
    val maxPeriod: Int,
    val maxSeqLen: Int,
    val kvRepeat: Int,
    val dimFeedforward: Int,
    val convLayout: Boolean,
    val sharedCrossAttn: Boolean = false,
)