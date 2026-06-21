// port-lint: source moshi-core/src/tts_streaming.rs
package io.github.kotlinmania.moshi

const val UNGENERATED_STREAMING: UInt = UInt.MAX_VALUE

data class TtsStreamingConfig(
    val acousticDelay: Int,
    val textPadToken: UInt,
    val textBosToken: UInt,
    val textEosToken: UInt,
    val textEopToken: UInt,
    val textStartToken: UInt,
    val textAudioDelayInTokens: Int,
    val maxConsecutivePads: Int,
    val extraSteps: Int,
    val speakerCondDurationS: Double,
    val speakerCondDim: Int,
    val speakerCondNSpeakers: Int,
) {
    companion object {
        fun v202501(): TtsStreamingConfig = TtsStreamingConfig(
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

enum class AllowedTokens {
    TEXT,
    PAD,
    PAD_OR_EPAD,
}