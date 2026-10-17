package io.github.kotlinmania.moshi

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
public enum class NormType {
    @SerialName("RmsNorm")
    RmsNorm,

    @SerialName("LayerNorm")
    LayerNorm,
}
