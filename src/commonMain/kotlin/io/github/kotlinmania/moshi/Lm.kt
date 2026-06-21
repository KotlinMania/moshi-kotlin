// port-lint: source moshi-core/src/lm.rs
package io.github.kotlinmania.moshi

data class DepFormerConfig(
    val transformer: TransformerConfig,
    val numSlices: Int,
    val lowRankEmbeddings: Int?,
)

data class ExtraHeadsConfig(
    val numHeads: Int,
    val dim: Int,
)

data class LmConfig(
    val transformer: TransformerConfig,
    val depformer: DepFormerConfig?,
    val textInVocabSize: Int,
    val textOutVocabSize: Int,
    val audioVocabSize: Int,
    val audioCodebooks: Int,
    val conditioners: ConditionerConfig?,
    val extraHeads: ExtraHeadsConfig?,
)

sealed class ConditionerConfig {
    data class Lut(val config: LutConfig) : ConditionerConfig()
    data class ContinuousAttribute(val config: ContinuousAttributeConfig) : ConditionerConfig()
}

data class LutConfig(
    val nBins: Int,
    val dim: Int,
    val possibleValues: List<String>,
)

data class ContinuousAttributeConfig(
    val dim: Int,
    val scaleFactor: Float,
    val maxPeriod: Float,
)