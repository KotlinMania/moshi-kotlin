package io.github.kotlinmania.moshi

import kotlin.test.Test
import kotlin.test.assertEquals

class TestByteArraySink : ByteArraySink {
    private val buffer = mutableListOf<Byte>()

    override fun write(bytes: ByteArray) {
        for (b in bytes) {
            buffer.add(b)
        }
    }

    fun toByteArray(): ByteArray = buffer.toByteArray()
}

class WavTest {
    @Test
    fun testSampleConversions() {
        val fSample = FloatSample(0.5f)
        assertEquals((0.5f * 32767f).toInt().toShort(), fSample.toI16())

        val dSample = DoubleSample(-0.5)
        assertEquals((-0.5 * 32767.0).toInt().toShort(), dSample.toI16())

        val sSample = ShortSample(1234)
        assertEquals(1234.toShort(), sSample.toI16())

        // Test clamping
        assertEquals(32767.toShort(), 2.0f.toI16())
        assertEquals((-32767).toShort(), (-2.0f).toI16())
        assertEquals(32767.toShort(), 2.0.toI16())
        assertEquals((-32767).toShort(), (-2.0).toI16())
    }

    @Test
    fun testWriteWavHeader() {
        val sink = TestByteArraySink()
        val sampleRate = 24000
        val chunkSize = 44
        val dataSize = 8

        writeWavHeader(sink, sampleRate, chunkSize, dataSize)
        val bytes = sink.toByteArray()

        assertEquals(44, bytes.size)
        // RIFF header
        assertEquals('R'.code.toByte(), bytes[0])
        assertEquals('I'.code.toByte(), bytes[1])
        assertEquals('F'.code.toByte(), bytes[2])
        assertEquals('F'.code.toByte(), bytes[3])

        // WAVE
        assertEquals('W'.code.toByte(), bytes[8])
        assertEquals('A'.code.toByte(), bytes[9])
        assertEquals('V'.code.toByte(), bytes[10])
        assertEquals('E'.code.toByte(), bytes[11])

        // fmt
        assertEquals('f'.code.toByte(), bytes[12])
        assertEquals('m'.code.toByte(), bytes[13])
        assertEquals('t'.code.toByte(), bytes[14])
        assertEquals(' '.code.toByte(), bytes[15])

        // data
        assertEquals('d'.code.toByte(), bytes[36])
        assertEquals('a'.code.toByte(), bytes[37])
        assertEquals('t'.code.toByte(), bytes[38])
        assertEquals('a'.code.toByte(), bytes[39])
    }

    @Test
    fun testWritePcmAsWavShorts() {
        val sink = TestByteArraySink()
        val samples = shortArrayOf(0, 1000, -1000, 32767)
        val sampleRate = 16000

        writePcmAsWav(sink, samples, sampleRate)
        val bytes = sink.toByteArray()

        assertEquals(44 + samples.size * 2, bytes.size)
    }

    @Test
    fun testWritePcmAsWavFloats() {
        val sink = TestByteArraySink()
        val samples = floatArrayOf(0f, 0.5f, -0.5f, 1.0f)
        val sampleRate = 24000

        writePcmAsWav(sink, samples, sampleRate)
        val bytes = sink.toByteArray()

        assertEquals(44 + samples.size * 2, bytes.size)
    }

    @Test
    fun testWritePcmAsWavSampleList() {
        val sink = TestByteArraySink()
        val samples =
            listOf(
                FloatSample(0.1f),
                DoubleSample(-0.2),
                ShortSample(500),
            )
        val sampleRate = 44100

        writePcmAsWav(sink, samples, sampleRate)
        val bytes = sink.toByteArray()

        assertEquals(44 + samples.size * 2, bytes.size)
    }
}
