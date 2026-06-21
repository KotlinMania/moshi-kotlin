// port-lint: source moshi-core/src/asr.rs
package io.github.kotlinmania.moshi

sealed class AsrMsg {
    data class Step(val stepIdx: Int, val prs: List<List<Float>>) : AsrMsg()
    data class Word(val tokens: List<Int>, val startTime: Double, val batchIdx: Int) : AsrMsg()
    data class EndWord(val stopTime: Double, val batchIdx: Int) : AsrMsg()
}