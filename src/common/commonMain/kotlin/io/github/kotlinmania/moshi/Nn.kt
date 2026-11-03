// port-lint: source nn.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.QTensor
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candlenn.Embedding
import io.github.kotlinmania.candlenn.Linear
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candlenn.embedding
import io.github.kotlinmania.candlenn.linearNoBias
import io.github.kotlinmania.candletransformers.QuantizedEmbedding
import io.github.kotlinmania.candletransformers.QuantizedLinear
import io.github.kotlinmania.candletransformers.QuantizedVarBuilder
import io.github.kotlinmania.candletransformers.quantizedLinearB

public sealed class MaybeQuantizedWeight {
    public data class Real(
        public val tensor: Tensor,
    ) : MaybeQuantizedWeight()

    public data class Quantized(
        public val qtensor: QTensor,
    ) : MaybeQuantizedWeight()

    public fun toTensor(dev: Device): Tensor =
        when (this) {
            is Real -> tensor
            is Quantized -> qtensor.dequantize(dev).getOrThrow()
        }
}

public fun matmulDtype(device: Device): DType =
    if (device.isCuda()) {
        DType.BF16
    } else {
        DType.F32
    }

public sealed class MaybeQuantizedVarBuilder {
    public data class Real(
        public val weights: VarBuilder,
    ) : MaybeQuantizedVarBuilder()

    public data class Quantized(
        public val weights: QuantizedVarBuilder,
    ) : MaybeQuantizedVarBuilder()

    public fun pp(s: String): MaybeQuantizedVarBuilder =
        when (this) {
            is Real -> Real(weights.pp(s))
            is Quantized -> Quantized(weights.pp(s))
        }

    public fun pp(s: Int): MaybeQuantizedVarBuilder = pp(s.toString())

    public fun get(
        s: Shape,
        path: String,
    ): MaybeQuantizedWeight =
        when (this) {
            is Real -> MaybeQuantizedWeight.Real(weights.get(s, path))
            is Quantized -> MaybeQuantizedWeight.Quantized(weights.get(s, path))
        }

    public fun get(
        s: Int,
        path: String,
    ): MaybeQuantizedWeight = get(Shape(s), path)

    public fun getAsTensor(
        s: Shape,
        path: String,
    ): Tensor =
        when (this) {
            is Real -> MaybeQuantizedWeight.Real(weights.get(s, path))
            is Quantized -> MaybeQuantizedWeight.Quantized(weights.get(s, path))
        }.toTensor(device())

    public fun getAsTensor(
        s: Int,
        path: String,
    ): Tensor = getAsTensor(Shape(s), path)

    public fun getUnquantized(
        s: Shape,
        path: String,
    ): Tensor =
        when (this) {
            is Real -> weights.get(s, path)
            is Quantized -> weights.get(s, path).dequantize(weights.device()).getOrThrow()
        }

    public fun getUnquantized(
        s: Int,
        path: String,
    ): Tensor = getUnquantized(Shape(s), path)

    public fun containsKey(name: String): Boolean =
        when (this) {
            is Real -> weights.containsTensor(name)
            is Quantized -> weights.containsKey(name)
        }

    public fun device(): Device =
        when (this) {
            is Real -> weights.device()
            is Quantized -> weights.device()
        }

    public fun dtype(): DType =
        when (this) {
            is Real -> weights.dtype()
            is Quantized -> DType.F32
        }
}

public sealed class MaybeQuantizedLinear : Module {
    public data class Real(
        public val module: Linear,
    ) : MaybeQuantizedLinear()

    public data class Quantized(
        public val module: QuantizedLinear,
    ) : MaybeQuantizedLinear()

    override fun forward(xs: Tensor): Tensor =
        when (this) {
            is Real -> module.forward(xs)
            is Quantized -> module.forward(xs)
        }

    public fun dtype(): DType =
        when (this) {
            is Real -> module.weight().dtype()
            is Quantized -> DType.F32
        }
}

public sealed class MaybeQuantizedEmbedding : Module {
    public data class Real(
        public val weights: Embedding,
    ) : MaybeQuantizedEmbedding()

    public data class Quantized(
        public val weights: QuantizedEmbedding,
    ) : MaybeQuantizedEmbedding()

    public companion object {
        public fun new(
            inVocabSize: Int,
            dim: Int,
            vb: MaybeQuantizedVarBuilder,
        ): MaybeQuantizedEmbedding =
            when (vb) {
                is MaybeQuantizedVarBuilder.Real -> Real(embedding(inVocabSize, dim, vb.weights))
                is MaybeQuantizedVarBuilder.Quantized -> Quantized(QuantizedEmbedding.new(inVocabSize, dim, vb.weights))
            }
    }

    public fun embeddings(): Tensor =
        when (this) {
            is Real -> weights.embeddings()
            is Quantized -> weights.embeddings()
        }

    public fun hiddenSize(): Int =
        when (this) {
            is Real -> weights.hiddenSize()
            is Quantized -> weights.embeddings().dim(1)
        }

    public fun dtype(): DType =
        when (this) {
            is Real -> weights.embeddings().dtype()
            is Quantized -> DType.F32
        }

    override fun forward(xs: Tensor): Tensor =
        when (this) {
            is Real -> weights.forward(xs)
            is Quantized -> weights.forward(xs)
        }
}

public fun linear(
    inD: Int,
    outD: Int,
    bias: Boolean,
    vb: MaybeQuantizedVarBuilder,
): MaybeQuantizedLinear =
    when (vb) {
        is MaybeQuantizedVarBuilder.Real -> {
            if (bias) {
                MaybeQuantizedLinear.Real(
                    io.github.kotlinmania.candlenn
                        .linear(inD, outD, vb.weights),
                )
            } else {
                MaybeQuantizedLinear.Real(linearNoBias(inD, outD, vb.weights))
            }
        }

        is MaybeQuantizedVarBuilder.Quantized -> {
            MaybeQuantizedLinear.Quantized(quantizedLinearB(inD, outD, bias, vb.weights))
        }
    }

public fun linearFrom(
    weight: MaybeQuantizedWeight,
    bias: Tensor?,
): MaybeQuantizedLinear =
    when (weight) {
        is MaybeQuantizedWeight.Real -> MaybeQuantizedLinear.Real(Linear.new(weight.tensor, bias))
        is MaybeQuantizedWeight.Quantized -> MaybeQuantizedLinear.Quantized(QuantizedLinear.fromArc(weight.qtensor, bias))
    }
