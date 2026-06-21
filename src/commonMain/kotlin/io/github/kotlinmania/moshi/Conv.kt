// port-lint: source moshi-core/src/conv.rs
package io.github.kotlinmania.moshi

enum class Norm {
    WEIGHT_NORM,
    SPECTRAL_NORM,
    TIME_GROUP_NORM,
}

enum class PadMode {
    CONSTANT,
    REFLECT,
    REPLICATE,
}