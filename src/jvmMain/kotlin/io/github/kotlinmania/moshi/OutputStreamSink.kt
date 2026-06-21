package io.github.kotlinmania.moshi

class OutputStreamSink(private val out: java.io.OutputStream) : ByteArraySink {
    override fun write(bytes: ByteArray) {
        out.write(bytes)
    }
}