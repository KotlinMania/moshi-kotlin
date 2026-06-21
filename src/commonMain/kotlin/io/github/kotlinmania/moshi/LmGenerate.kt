// port-lint: source moshi-core/src/lm_generate.rs
package io.github.kotlinmania.moshi

sealed class Token {
    data class Set(val value: UInt) : Token()
    data object Ungenerated : Token()
    data object LiteralZero : Token()
}

data class LmGenerateConfig(
    val audioDelays: List<Int>,
    val audioVocabSize: Int,
    val textPadToken: UInt,
    val textEopToken: UInt,
    val textStartToken: UInt,
) {
    fun audioPadToken(): UInt = (audioVocabSize - 1).toUInt()

    fun audioCodebooks(): Int = audioDelays.size

    fun maxAudioDelay(): Int = audioDelays.maxOrNull() ?: 0
}