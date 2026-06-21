// port-lint: source moshi-core/src/lm_generate_multistream.rs
package io.github.kotlinmania.moshi

const val UNGENERATED: UInt = UInt.MAX_VALUE

data class MultiStreamConfig(
    val generatedAudioCodebooks: Int,
    val inputAudioCodebooks: Int,
    val audioVocabSize: Int,
    val acousticDelay: Int,
    val textPadToken: UInt,
    val textEopToken: UInt,
    val textStartToken: UInt,
) {
    companion object {
        fun v01(): MultiStreamConfig = MultiStreamConfig(
            generatedAudioCodebooks = 8,
            inputAudioCodebooks = 8,
            audioVocabSize = 2049,
            acousticDelay = 2,
            textEopToken = 0u,
            textPadToken = 3u,
            textStartToken = 32000u,
        )

        fun v01TwoWays(): MultiStreamConfig = MultiStreamConfig(
            generatedAudioCodebooks = 16,
            inputAudioCodebooks = 0,
            audioVocabSize = 2049,
            acousticDelay = 2,
            textEopToken = 0u,
            textPadToken = 3u,
            textStartToken = 32000u,
        )

        fun v01OneWay(): MultiStreamConfig = MultiStreamConfig(
            generatedAudioCodebooks = 8,
            inputAudioCodebooks = 0,
            audioVocabSize = 2049,
            acousticDelay = 2,
            textEopToken = 0u,
            textPadToken = 3u,
            textStartToken = 32000u,
        )
    }

    fun audioPadToken(): UInt = (audioVocabSize - 1).toUInt()

    fun totalAudioCodebooks(): Int = generatedAudioCodebooks + inputAudioCodebooks
}