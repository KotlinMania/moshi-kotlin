// port-lint: source moshi-core/src/wav.rs
package io.github.kotlinmania.moshi

fun Float.toI16(): Short = (this.coerceIn(-1f, 1f) * 32767f).toInt().toShort()

fun Double.toI16(): Short = (this.coerceIn(-1.0, 1.0) * 32767.0).toInt().toShort()

interface ByteArraySink {
    fun write(bytes: ByteArray)
    fun writeIntLittleEndian(value: Int) {
        write(
            byteArrayOf(
                (value and 0xFF).toByte(),
                ((value shr 8) and 0xFF).toByte(),
                ((value shr 16) and 0xFF).toByte(),
                ((value shr 24) and 0xFF).toByte(),
            ),
        )
    }

    fun writeShortLittleEndian(value: Short) {
        write(
            byteArrayOf(
                (value.toInt() and 0xFF).toByte(),
                ((value.toInt() shr 8) and 0xFF).toByte(),
            ),
        )
    }
}

fun writeWavHeader(
    sink: ByteArraySink,
    sampleRate: Int,
    chunkSize: Int,
    dataSize: Int,
) {
    val nChannels: Short = 1
    val bitsPerSample: Short = 16
    val byteRate = sampleRate * nChannels.toInt() * (bitsPerSample.toInt() / 8)
    val blockAlign = (nChannels.toInt() * (bitsPerSample.toInt() / 8)).toShort()

    sink.write(byteArrayOf('R'.code.toByte(), 'I'.code.toByte(), 'F'.code.toByte(), 'F'.code.toByte()))
    sink.writeIntLittleEndian(chunkSize)
    sink.write(byteArrayOf('W'.code.toByte(), 'A'.code.toByte(), 'V'.code.toByte(), 'E'.code.toByte()))

    sink.write(byteArrayOf('f'.code.toByte(), 'm'.code.toByte(), 't'.code.toByte(), ' '.code.toByte()))
    sink.writeIntLittleEndian(16)
    sink.writeShortLittleEndian(1)
    sink.writeShortLittleEndian(nChannels)
    sink.writeIntLittleEndian(sampleRate)
    sink.writeIntLittleEndian(byteRate)
    sink.writeShortLittleEndian(blockAlign)
    sink.writeShortLittleEndian(bitsPerSample)

    sink.write(byteArrayOf('d'.code.toByte(), 'a'.code.toByte(), 't'.code.toByte(), 'a'.code.toByte()))
    sink.writeIntLittleEndian(dataSize)
}

fun writePcmInWav(sink: ByteArraySink, samples: ShortArray): Int {
    for (sample in samples) {
        sink.writeShortLittleEndian(sample)
    }
    return samples.size * 2
}

fun writePcmAsWav(sink: ByteArraySink, samples: ShortArray, sampleRate: Int) {
    val chunkSize = 12u + 24u + samples.size.toUInt() * 2u
    val dataSize = samples.size.toUInt() * 2u
    writeWavHeader(sink, sampleRate, chunkSize.toInt(), dataSize.toInt())
    writePcmInWav(sink, samples)
}