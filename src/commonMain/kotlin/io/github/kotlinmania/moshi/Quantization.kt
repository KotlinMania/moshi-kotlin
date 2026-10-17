// port-lint: source quantization.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CustomOp2
import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.applyOp2
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candle.i
import io.github.kotlinmania.candlenn.Conv1d
import io.github.kotlinmania.candlenn.Conv1dConfig
import io.github.kotlinmania.candlenn.Linear
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candlenn.conv1dNoBias

private object CodebookEncode : CustomOp2 {
    override fun name(): String = "cb"

    override fun cpuFwd(
        s1: CpuStorage,
        l1: Layout,
        s2: CpuStorage,
        l2: Layout,
    ): Pair<CpuStorage, Shape> {
        val lhsDims = l1.dims()
        if (lhsDims.size != 2) {
            bail("CodebookEncode expects 2D lhs, got $l1")
        }
        val lhsDim1 = lhsDims[0]
        val lhsDim2 = lhsDims[1]
        val rhsDims = l2.dims()
        if (rhsDims.size != 2) {
            bail("CodebookEncode expects 2D rhs, got $l2")
        }
        val rhsDim1 = rhsDims[0]
        val rhsDim2 = rhsDims[1]
        if (lhsDim2 != rhsDim2) {
            bail("CodebookEncode, mismatch on last dim, $l1 $l2")
        }
        if (lhsDim2 == 0) {
            bail("CodebookEncode, empty last dim $l1")
        }
        val lhsOffsets =
            l1.contiguousOffsets()
                ?: bail("CodebookEncode, lhs has to be contiguous, got $l1")
        val rhsOffsets =
            l2.contiguousOffsets()
                ?: bail("CodebookEncode, rhs has to be contiguous, got $l2")
        val lhsSlice =
            when (s1) {
                is CpuStorage.F32 -> s1.data
                else -> bail("CodebookEncode expects F32 lhs")
            }
        val rhsSlice =
            when (s2) {
                is CpuStorage.F32 -> s2.data
                else -> bail("CodebookEncode expects F32 rhs")
            }
        val lhsStart = lhsOffsets.first
        val rhsStart = rhsOffsets.first
        val dst = UIntArray(lhsDim1)
        for (idx1 in 0 until lhsDim1) {
            var whereMin = 0
            var minDist = Float.POSITIVE_INFINITY
            val lhsBase = lhsStart + idx1 * lhsDim2
            for (idx2 in 0 until rhsDim1) {
                val rhsBase = rhsStart + idx2 * rhsDim2
                var dist = 0.0f
                for (k in 0 until lhsDim2) {
                    val diff = lhsSlice[lhsBase + k] - rhsSlice[rhsBase + k]
                    dist += diff * diff
                }
                if (dist < minDist) {
                    minDist = dist
                    whereMin = idx2
                }
            }
            dst[idx1] = whereMin.toUInt()
        }
        return CpuStorage.U32(dst) to Shape(lhsDim1)
    }
}

public class EuclideanCodebook(
    private val initialized: Tensor,
    private val clusterUsage: Tensor,
    private val embeddingSum: Tensor,
    private val embedding: Tensor,
    private val c2: Tensor,
    private val epsilon: Double,
    private val dim: Int,
) {
    public companion object {
        public fun new(
            dim: Int,
            codebookSize: Int,
            vb: VarBuilder,
        ): EuclideanCodebook {
            val epsilon = 1e-5
            val initialized = vb.get(1, "_initialized")
            val clusterUsage = vb.get(codebookSize, "cluster_usage")
            val embeddingSum = vb.get(Shape(codebookSize, dim), "embedding_sum")
            val clampedClusterUsage = clusterUsage.clamp(epsilon, Double.POSITIVE_INFINITY).unsqueeze(1)
            val embedding = embeddingSum.broadcastDiv(clampedClusterUsage)
            val c2 = embedding.mul(embedding).sum(embedding.rank() - 1).affine(0.5, 0.0)
            return EuclideanCodebook(
                initialized = initialized,
                clusterUsage = clusterUsage,
                embeddingSum = embeddingSum,
                embedding = embedding,
                c2 = c2,
                epsilon = epsilon,
                dim = dim,
            )
        }
    }

    public fun encodeVerySlow(xs: Tensor): Tensor {
        val targetDims = xs.dims().dropLast(1)
        val flatXs = xs.flattenTo(D.Minus2)
        flatXs.dims2()
        val clampedClusterUsage = clusterUsage.clamp(epsilon, Double.POSITIVE_INFINITY).unsqueeze(1)
        val emb = embeddingSum.broadcastDiv(clampedClusterUsage)
        val diff = flatXs.unsqueeze(1).broadcastSub(emb.unsqueeze(0))
        val dists = diff.sqr().sum(diff.rank() - 1)
        val codes = dists.argmin(dists.rank() - 1)
        return codes.reshape(Shape(targetDims))
    }

    public fun encodeSlow(xs: Tensor): Tensor {
        val targetDims = xs.dims().dropLast(1)
        val flatXs = xs.flattenTo(D.Minus2)
        flatXs.dims2()
        val dotProd = flatXs.matmul(embedding.t())
        val sub = c2.broadcastSub(dotProd)
        val codes = sub.argmin(sub.rank() - 1)
        return codes.reshape(Shape(targetDims))
    }

    public fun encode(xs: Tensor): Tensor {
        val targetDims = xs.dims().dropLast(1)
        val flatXs = xs.flattenTo(D.Minus2)
        flatXs.dims2()
        val codes = flatXs.applyOp2(embedding, CodebookEncode)
        return codes.reshape(Shape(targetDims))
    }

    public fun decode(indexes: Tensor): Tensor {
        val finalDims = indexes.dims() + dim
        val flatIndexes = indexes.flattenAll()
        val values = embedding.indexSelect(flatIndexes, 0)
        return values.reshape(Shape(finalDims))
    }
}

public class VectorQuantization(
    private val projectIn: Linear?,
    private val projectOut: Linear?,
    private val codebook: EuclideanCodebook,
) {
    public companion object {
        public fun new(
            dim: Int,
            codebookSize: Int,
            codebookDim: Int?,
            vb: VarBuilder,
        ): VectorQuantization {
            val resolvedCodebookDim = codebookDim ?: dim
            val (projectIn, projectOut) =
                if (resolvedCodebookDim == dim) {
                    null to null
                } else {
                    val pIn =
                        io.github.kotlinmania.candlenn
                            .linear(dim, resolvedCodebookDim, vb.pp("project_in"))
                    val pOut =
                        io.github.kotlinmania.candlenn
                            .linear(resolvedCodebookDim, dim, vb.pp("project_out"))
                    pIn to pOut
                }
            val codebook = EuclideanCodebook.new(resolvedCodebookDim, codebookSize, vb.pp("_codebook"))
            return VectorQuantization(projectIn, projectOut, codebook)
        }
    }

    public fun encode(xs: Tensor): Tensor {
        val transposed = xs.t()
        val projected = if (projectIn != null) transposed.apply(projectIn) else transposed
        return codebook.encodeSlow(projected)
    }

    public fun decode(codes: Tensor): Tensor {
        val quantized = codebook.decode(codes)
        val projected = if (projectOut != null) quantized.apply(projectOut) else quantized
        return projected.t()
    }
}

public class ResidualVectorQuantization(
    private val layers: List<VectorQuantization>,
) {
    public companion object {
        public fun new(
            nQ: Int,
            dim: Int,
            codebookSize: Int,
            codebookDim: Int?,
            vb: VarBuilder,
        ): ResidualVectorQuantization {
            val vbLayers = vb.pp("layers")
            val layers =
                List(nQ) { i ->
                    VectorQuantization.new(dim, codebookSize, codebookDim, vbLayers.pp(i))
                }
            return ResidualVectorQuantization(layers)
        }
    }

    public fun encode(xs: Tensor): Tensor {
        val codes = ArrayList<Tensor>(layers.size)
        var residual = xs
        for (layer in layers) {
            val indices = layer.encode(residual)
            val quantized = layer.decode(indices)
            residual = residual.sub(quantized)
            codes.add(indices)
        }
        return Tensor.stack(codes, 0)
    }

    public fun decode(xs: Tensor): Tensor {
        if (layers.isEmpty()) {
            bail("empty layers in ResidualVectorQuantization")
        }
        if (layers.size != xs.dim(0)) {
            bail("mismatch between the number of layers ${layers.size} and the code shape ${xs.shape()}")
        }
        var quantized = layers[0].decode(xs.i(0))
        for (i in 1 until layers.size) {
            val layerCodes = xs.i(i)
            quantized = quantized.add(layers[i].decode(layerCodes))
        }
        return quantized
    }
}

public class ResidualVectorQuantizer(
    private val vq: ResidualVectorQuantization,
    private val inputProj: Conv1d?,
    private val outputProj: Conv1d?,
) {
    public companion object {
        public fun new(
            dim: Int,
            inputDim: Int?,
            outputDim: Int?,
            nQ: Int,
            bins: Int,
            forceProjection: Boolean,
            vb: VarBuilder,
        ): ResidualVectorQuantizer {
            val resolvedInputDim = inputDim ?: dim
            val resolvedOutputDim = outputDim ?: dim
            val inputProj =
                if (resolvedInputDim == dim && !forceProjection) {
                    null
                } else {
                    conv1dNoBias(
                        resolvedInputDim,
                        dim,
                        1,
                        Conv1dConfig.default(),
                        vb.pp("input_proj"),
                    )
                }
            val outputProj =
                if (resolvedOutputDim == dim && !forceProjection) {
                    null
                } else {
                    conv1dNoBias(
                        dim,
                        resolvedOutputDim,
                        1,
                        Conv1dConfig.default(),
                        vb.pp("output_proj"),
                    )
                }
            val vq =
                ResidualVectorQuantization.new(
                    nQ = nQ,
                    dim = dim,
                    codebookSize = bins,
                    codebookDim = null,
                    vb = vb.pp("vq"),
                )
            return ResidualVectorQuantizer(vq, inputProj, outputProj)
        }
    }

    public fun encode(xs: Tensor): Tensor {
        val projected = if (inputProj != null) xs.apply(inputProj) else xs
        val codes = vq.encode(projected)
        return codes.transpose(0, 1)
    }

    public fun decode(codes: Tensor): Tensor {
        // codes is [B, K, T], with T frames, K nb of codebooks, vq.decode expects [K, B, T].
        val transposed = codes.transpose(0, 1)
        val quantized = vq.decode(transposed)
        return if (outputProj != null) quantized.apply(outputProj) else quantized
    }
}

/**
 * We do not use any codebook offset at the moment. When reconstructing the codes, we could just
 * concatenate the indexes.
 */
public class SplitResidualVectorQuantizer(
    private val rvqFirst: ResidualVectorQuantizer,
    private val rvqRest: ResidualVectorQuantizer,
    private val nQ: Int,
) {
    public companion object {
        public fun new(
            dim: Int,
            inputDim: Int?,
            outputDim: Int?,
            nQ: Int,
            bins: Int,
            vb: VarBuilder,
        ): SplitResidualVectorQuantizer {
            val rvqFirst =
                ResidualVectorQuantizer.new(
                    dim = dim,
                    inputDim = inputDim,
                    outputDim = outputDim,
                    nQ = 1,
                    bins = bins,
                    forceProjection = true,
                    vb = vb.pp("rvq_first"),
                )
            val rvqRest =
                ResidualVectorQuantizer.new(
                    dim = dim,
                    inputDim = inputDim,
                    outputDim = outputDim,
                    nQ = nQ - 1,
                    bins = bins,
                    forceProjection = true,
                    vb = vb.pp("rvq_rest"),
                )
            return SplitResidualVectorQuantizer(rvqFirst, rvqRest, nQ)
        }
    }

    public fun encode(xs: Tensor): Tensor {
        val codes = rvqFirst.encode(xs)
        return if (nQ > 1) {
            // We encode xs again here rather than the residual. The decomposition is not
            // hierarchical but rather having semantic tokens for rvqFirst and the acoustic tokens
            // for rvqRest.
            val restCodes = rvqRest.encode(xs)
            Tensor.cat(listOf(codes, restCodes), 1)
        } else {
            codes
        }
    }

    public fun decode(codes: Tensor): Tensor {
        // codes is [B, K, T], with T frames, K nb of codebooks.
        val firstCodes = codes.narrow(1, 0, 1)
        var quantized = rvqFirst.decode(firstCodes)
        if (nQ > 1) {
            val kLen = codes.dim(1)
            val restCodes = codes.narrow(1, 1, kLen - 1)
            quantized = quantized.add(rvqRest.decode(restCodes))
        }
        return quantized
    }
}
