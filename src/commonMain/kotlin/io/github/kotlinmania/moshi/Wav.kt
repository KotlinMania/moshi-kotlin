// port-lint: source wav.rs
package io.github.kotlinmania.moshi

public interface Sample {
    public fun toI16(): Short
}

public fun Float.toI16(): Short = (this.coerceIn(-1.0f, 1.0f) * 32767.0f).toInt().toShort()

public fun Double.toI16(): Short = (this.coerceIn(-1.0, 1.0) * 32767.0).toInt().toShort()

public fun Short.toI16(): Short = this

public class FloatSample(
    public val value: Float,
) : Sample {
    override fun toI16(): Short = value.toI16()
}

public class DoubleSample(
    public val value: Double,
) : Sample {
    override fun toI16(): Short = value.toI16()
}

public class ShortSample(
    public val value: Short,
) : Sample {
    override fun toI16(): Short = value
}

public interface ByteArraySink {
    public fun write(bytes: ByteArray)

    public fun writeIntLittleEndian(value: Int) {
        write(
            byteArrayOf(
                (value and 0xFF).toByte(),
                ((value shr 8) and 0xFF).toByte(),
                ((value shr 16) and 0xFF).toByte(),
                ((value shr 24) and 0xFF).toByte(),
            ),
        )
    }

    public fun writeShortLittleEndian(value: Short) {
        write(
            byteArrayOf(
                (value.toInt() and 0xFF).toByte(),
                ((value.toInt() shr 8) and 0xFF).toByte(),
            ),
        )
    }
}

public fun writeWavHeader(
    w: ByteArraySink,
    sampleRate: Int,
    chunkSize: Int,
    dataSize: Int,
) {
    val nChannels: Short = 1
    val bitsPerSample: Short = 16
    val byteRate = sampleRate * nChannels.toInt() * (bitsPerSample.toInt() / 8)
    val blockAlign = (nChannels.toInt() * (bitsPerSample.toInt() / 8)).toShort()

    w.write(byteArrayOf('R'.code.toByte(), 'I'.code.toByte(), 'F'.code.toByte(), 'F'.code.toByte()))
    w.writeIntLittleEndian(chunkSize)
    w.write(byteArrayOf('W'.code.toByte(), 'A'.code.toByte(), 'V'.code.toByte(), 'E'.code.toByte()))

    w.write(byteArrayOf('f'.code.toByte(), 'm'.code.toByte(), 't'.code.toByte(), ' '.code.toByte()))
    w.writeIntLittleEndian(16)
    w.writeShortLittleEndian(1)
    w.writeShortLittleEndian(nChannels)
    w.writeIntLittleEndian(sampleRate)
    w.writeIntLittleEndian(byteRate)
    w.writeShortLittleEndian(blockAlign)
    w.writeShortLittleEndian(bitsPerSample)

    w.write(byteArrayOf('d'.code.toByte(), 'a'.code.toByte(), 't'.code.toByte(), 'a'.code.toByte()))
    w.writeIntLittleEndian(dataSize)
}

public fun writePcmInWav(
    w: ByteArraySink,
    samples: ShortArray,
): Int {
    for (sample in samples) {
        w.writeShortLittleEndian(sample)
    }
    return samples.size * 2
}

public fun writePcmInWav(
    w: ByteArraySink,
    samples: FloatArray,
): Int {
    for (sample in samples) {
        w.writeShortLittleEndian(sample.toI16())
    }
    return samples.size * 2
}

public fun writePcmInWav(
    w: ByteArraySink,
    samples: DoubleArray,
): Int {
    for (sample in samples) {
        w.writeShortLittleEndian(sample.toI16())
    }
    return samples.size * 2
}

public fun <S : Sample> writePcmInWav(
    w: ByteArraySink,
    samples: List<S>,
): Int {
    for (sample in samples) {
        w.writeShortLittleEndian(sample.toI16())
    }
    return samples.size * 2
}

public fun writePcmAsWav(
    w: ByteArraySink,
    samples: ShortArray,
    sampleRate: Int,
) {
    val chunkSize = 12u + 24u + samples.size.toUInt() * 2u
    val dataSize = samples.size.toUInt() * 2u
    writeWavHeader(w, sampleRate, chunkSize.toInt(), dataSize.toInt())
    writePcmInWav(w, samples)
}

public fun writePcmAsWav(
    w: ByteArraySink,
    samples: FloatArray,
    sampleRate: Int,
) {
    val chunkSize = 12u + 24u + samples.size.toUInt() * 2u
    val dataSize = samples.size.toUInt() * 2u
    writeWavHeader(w, sampleRate, chunkSize.toInt(), dataSize.toInt())
    writePcmInWav(w, samples)
}

public fun writePcmAsWav(
    w: ByteArraySink,
    samples: DoubleArray,
    sampleRate: Int,
) {
    val chunkSize = 12u + 24u + samples.size.toUInt() * 2u
    val dataSize = samples.size.toUInt() * 2u
    writeWavHeader(w, sampleRate, chunkSize.toInt(), dataSize.toInt())
    writePcmInWav(w, samples)
}

public fun <S : Sample> writePcmAsWav(
    w: ByteArraySink,
    samples: List<S>,
    sampleRate: Int,
) {
    val chunkSize = 12u + 24u + samples.size.toUInt() * 2u
    val dataSize = samples.size.toUInt() * 2u
    writeWavHeader(w, sampleRate, chunkSize.toInt(), dataSize.toInt())
    writePcmInWav(w, samples)
}
