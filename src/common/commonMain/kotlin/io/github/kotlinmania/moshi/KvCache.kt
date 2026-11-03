// port-lint: source kv_cache.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candlenn.RotatingCache
import io.github.kotlinmania.candlenn.RotatingKvCache

public data class KeyValuePair(
    public val k: Tensor,
    public val v: Tensor,
)

public class IndicesAndMask(
    internal val indices: Tensor,
    private val mask: Tensor,
) {
    public fun indices(): Tensor = indices

    public fun mask(): Tensor = mask
}

public class ScatteredKvCache(
    private val k: Tensor,
    private val v: Tensor,
    private val context: Int,
) {
    public fun append(
        k: Tensor,
        v: Tensor,
        iam: IndicesAndMask,
    ): KeyValuePair {
        if (context <= k.dim(2)) {
            return KeyValuePair(k, v)
        }
        val indices =
            iam.indices
                .unsqueeze(2)
                .unsqueeze(1)
                .broadcastAs(k.shape())
                .contiguous()
        this.k.scatterSet(indices, k, 2)
        this.v.scatterSet(indices, v, 2)
        return KeyValuePair(this.k, this.v)
    }

    public fun k(): Tensor = k

    public fun v(): Tensor = v

    public fun copy(): ScatteredKvCache =
        ScatteredKvCache(
            k = k.copy(),
            v = v.copy(),
            context = context,
        )
}

public class ScatteredCacheBuilder(
    private val context: Int,
    private val positions: IntArray,
    private val indices: IntArray,
    private val dtype: DType,
    private val device: Device,
) {
    public companion object {
        public fun new(
            batchSize: Int,
            context: Int,
            dtype: DType,
            device: Device,
        ): ScatteredCacheBuilder =
            ScatteredCacheBuilder(
                context = context,
                positions = IntArray(batchSize),
                indices = IntArray(batchSize),
                dtype = dtype,
                device = device,
            )
    }

    public fun makeCache(
        numHeads: Int,
        headDim: Int,
    ): ScatteredKvCache {
        val batchSize = batchSize()
        val shape = Shape(batchSize, numHeads, context, headDim)
        val k = Tensor.zeros(shape, dtype, device())
        val v = Tensor.zeros(shape, dtype, device())
        return ScatteredKvCache(k, v, context)
    }

    public fun positions(): IntArray = positions.copyOf()

    public fun reset() {
        positions.fill(0)
        indices.fill(0)
    }

    public fun batchSize(): Int = positions.size

    public fun resetBatchIndex(batchIndex: Int) {
        positions[batchIndex] = 0
        indices[batchIndex] = 0
    }

    public fun indicesAndMask(
        seqLen: Int,
        batchMask: List<Boolean>,
    ): IndicesAndMask {
        if (context <= seqLen) {
            return indicesAndMaskAbs(seqLen, batchMask)
        }
        val bSize = batchSize()
        val attentionMasks = FloatArray(bSize * seqLen * context)
        val cacheIndices = UIntArray(bSize * seqLen)
        for (batchI in 0 until bSize) {
            val active = batchMask[batchI]
            if (!active) {
                val curIdx = indices[batchI].toUInt()
                for (seqI in 0 until seqLen) {
                    cacheIndices[batchI * seqLen + seqI] = curIdx
                }
            } else {
                val startIndex = indices[batchI]
                val startPos = positions[batchI]
                val allPos = IntArray(context) { Int.MAX_VALUE }
                if (startPos < context) {
                    for (i in 0 until startPos) {
                        allPos[i] = i
                    }
                } else {
                    val offset = startPos - startIndex
                    for (i in 0 until context) {
                        allPos[i] = if (i < startIndex) i + offset else i + offset - context
                    }
                }
                for (seqI in 0 until seqLen) {
                    val index = indices[batchI]
                    allPos[index] = seqI + startPos
                    cacheIndices[batchI * seqLen + seqI] = index.toUInt()
                    indices[batchI] += 1
                    positions[batchI] += 1
                    if (indices[batchI] >= context) {
                        indices[batchI] = 0
                    }
                }
                for (seqI in 0 until seqLen) {
                    val myPos = seqI + startPos
                    val rowBase = (batchI * seqLen + seqI) * context
                    for (c in 0 until context) {
                        attentionMasks[rowBase + c] =
                            if (allPos[c] <= myPos) 0.0f else Float.NEGATIVE_INFINITY
                    }
                }
            }
        }
        val mask =
            Tensor
                .fromSlice(attentionMasks, Shape(bSize, 1, seqLen, context), device())
                .toDtype(dtype)
        val indicesTensor = Tensor.fromSlice(cacheIndices, Shape(bSize, seqLen), device())
        return IndicesAndMask(indicesTensor, mask)
    }

    public fun device(): Device = device

    private fun indicesAndMaskAbs(
        seqLen: Int,
        batchMask: List<Boolean>,
    ): IndicesAndMask {
        val mask = getMaskAbs(seqLen, seqLen)
        val bSize = batchSize()
        val cacheIndices = UIntArray(bSize * seqLen)
        for (batchI in 0 until bSize) {
            val active = batchMask[batchI]
            if (!active) {
                val curIdx = indices[batchI].toUInt()
                for (seqI in 0 until seqLen) {
                    cacheIndices[batchI * seqLen + seqI] = curIdx
                }
            } else {
                for (seqI in 0 until seqLen) {
                    val index = indices[batchI]
                    cacheIndices[batchI * seqLen + seqI] = index.toUInt()
                    indices[batchI] += 1
                    positions[batchI] += 1
                    if (indices[batchI] >= context) {
                        indices[batchI] = 0
                    }
                }
            }
        }
        val indicesTensor = Tensor.fromSlice(cacheIndices, Shape(bSize, seqLen), device())
        return IndicesAndMask(indicesTensor, mask)
    }

    private fun getMaskAbs(
        size1: Int,
        size2: Int,
    ): Tensor {
        val mask = FloatArray(size1 * size2)
        for (i in 0 until size1) {
            for (j in 0 until size2) {
                mask[i * size2 + j] =
                    if (size1 + j > size2 + i || size1 + j + context < size2 + i) {
                        Float.NEGATIVE_INFINITY
                    } else {
                        0.0f
                    }
            }
        }
        return Tensor.fromSlice(mask, Shape(size1, size2), device())
    }
}

public class KvCache(
    private val cache: RotatingKvCache,
) {
    public companion object {
        public fun new(
            dim: Int,
            maxSeqLen: Int,
        ): KvCache = KvCache(RotatingKvCache.new(dim, maxSeqLen))
    }

    public fun currentSeqLen(): Int = cache.currentSeqLen()

    public fun reset() {
        cache.reset()
    }

    public fun append(
        key: Tensor,
        value: Tensor,
    ): KeyValuePair {
        val pair = cache.append(key, value)
        return KeyValuePair(pair.first, pair.second)
    }

    public fun positions(seqLen: Int): List<Int> = cache.positions(seqLen)

    public fun copy(): KvCache {
        val kSrc = cache.kCache()
        val vSrc = cache.vCache()
        val kCopy =
            RotatingCache(
                allData = kSrc.allData(),
                dim = kSrc.dim(),
                offset = kSrc.offset(),
                currentSeqLen = kSrc.currentSeqLen(),
                maxSeqLen = kSrc.maxSeqLen(),
            )
        val vCopy =
            RotatingCache(
                allData = vSrc.allData(),
                dim = vSrc.dim(),
                offset = vSrc.offset(),
                currentSeqLen = vSrc.currentSeqLen(),
                maxSeqLen = vSrc.maxSeqLen(),
            )
        return KvCache(RotatingKvCache(kCopy, vCopy))
    }
}
