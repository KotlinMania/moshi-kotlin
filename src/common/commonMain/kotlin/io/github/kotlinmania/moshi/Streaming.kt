// port-lint: source streaming.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import kotlin.math.min

public class MaskInner(
    public val cpu: List<Boolean>,
    public val mask: Tensor,
)

public class StreamMask(
    private val inner: MaskInner? = null,
) {
    public companion object {
        public fun empty(): StreamMask = StreamMask(null)

        public fun new(
            cpu: List<Boolean>,
            device: Device,
        ): StreamMask {
            val maskBytes = UByteArray(cpu.size) { idx -> if (cpu[idx]) 1u else 0u }
            val mask = Tensor.new(maskBytes, Shape(cpu.size), device)
            return StreamMask(MaskInner(cpu, mask))
        }
    }

    public fun isActive(batchIdx: Int): Boolean {
        val cpuMask = cpu() ?: return true
        return cpuMask[batchIdx]
    }

    public fun isEmpty(): Boolean = inner == null

    public fun shape(): Shape? = inner?.mask?.shape()

    public fun asOption(): Tensor? = inner?.mask

    public fun cpu(): List<Boolean>? = inner?.cpu

    override fun toString(): String =
        if (inner != null) {
            inner.mask.shape().toString()
        } else {
            "Empty"
        }
}

public data class StreamTensorSplit(
    public val lhs: StreamTensor,
    public val rhs: StreamTensor,
)

public class StreamTensor(
    private var tensor: Tensor? = null,
) {
    public companion object {
        public fun empty(): StreamTensor = StreamTensor(null)

        public fun fromTensor(tensor: Tensor): StreamTensor = StreamTensor(tensor)

        public fun cat2(
            lhs: StreamTensor,
            rhs: StreamTensor,
            dim: D,
        ): StreamTensor {
            val l = lhs.tensor
            val r = rhs.tensor
            val xs =
                if (l != null && r != null) {
                    Tensor.cat(listOf(l, r), dim)
                } else {
                    l ?: r
                }
            return StreamTensor(xs)
        }

        public fun cat2(
            lhs: StreamTensor,
            rhs: StreamTensor,
            dim: Int,
        ): StreamTensor {
            val l = lhs.tensor
            val r = rhs.tensor
            val xs =
                if (l != null && r != null) {
                    Tensor.cat(listOf(l, r), dim)
                } else {
                    l ?: r
                }
            return StreamTensor(xs)
        }
    }

    public fun isEmpty(): Boolean = tensor == null

    public fun shape(): Shape? = tensor?.shape()

    public fun cat2(
        rhs: StreamTensor,
        dim: D,
    ): StreamTensor = Companion.cat2(this, rhs, dim)

    public fun cat2(
        rhs: StreamTensor,
        dim: Int,
    ): StreamTensor = Companion.cat2(this, rhs, dim)

    public fun seqLen(dim: D): Int = tensor?.dim(dim) ?: 0

    public fun seqLen(dim: Int): Int = tensor?.dim(dim) ?: 0

    public fun reset() {
        tensor = null
    }

    public fun narrow(
        dim: D,
        offset: Int,
        len: Int,
    ): StreamTensor {
        val current = tensor ?: return StreamTensor(null)
        val seqLen = current.dim(dim)
        return if (seqLen <= offset) {
            StreamTensor(null)
        } else {
            StreamTensor(current.narrow(dim, offset, min(len, seqLen - offset)))
        }
    }

    public fun narrow(
        dim: Int,
        offset: Int,
        len: Int,
    ): StreamTensor {
        val current = tensor ?: return StreamTensor(null)
        val seqLen = current.dim(dim)
        return if (seqLen <= offset) {
            StreamTensor(null)
        } else {
            StreamTensor(current.narrow(dim, offset, min(len, seqLen - offset)))
        }
    }

    /**
     * Splits the Streaming Tensor on the time axis `dim` with the first `lhsLen` elements
     * returned in the first output and the remaining in the second output.
     */
    public fun split(
        dim: D,
        lhsLen: Int,
    ): StreamTensorSplit {
        val current = tensor ?: return StreamTensorSplit(empty(), empty())
        val seqLen = current.dim(dim)
        val clampedLhsLen = min(seqLen, lhsLen)
        if (clampedLhsLen == 0) {
            return StreamTensorSplit(empty(), fromTensor(current))
        }
        val lhs = fromTensor(current.narrow(dim, 0, clampedLhsLen))
        val rhsLen = seqLen - clampedLhsLen
        val rhs =
            if (rhsLen == 0) {
                empty()
            } else {
                fromTensor(current.narrow(dim, clampedLhsLen, rhsLen))
            }
        return StreamTensorSplit(lhs, rhs)
    }

    public fun split(
        dim: Int,
        lhsLen: Int,
    ): StreamTensorSplit {
        val current = tensor ?: return StreamTensorSplit(empty(), empty())
        val seqLen = current.dim(dim)
        val clampedLhsLen = min(seqLen, lhsLen)
        if (clampedLhsLen == 0) {
            return StreamTensorSplit(empty(), fromTensor(current))
        }
        val lhs = fromTensor(current.narrow(dim, 0, clampedLhsLen))
        val rhsLen = seqLen - clampedLhsLen
        val rhs =
            if (rhsLen == 0) {
                empty()
            } else {
                fromTensor(current.narrow(dim, clampedLhsLen, rhsLen))
            }
        return StreamTensorSplit(lhs, rhs)
    }

    public fun asOption(): Tensor? = tensor

    public fun apply(m: Module?): StreamTensor {
        if (m == null) return StreamTensor(tensor)
        val current = tensor ?: return empty()
        return fromTensor(current.apply(m))
    }

    override fun toString(): String = tensor?.shape()?.toString() ?: "Empty"
}

public interface StreamingModule {
    public fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor

    public fun resetState()
}

public enum class BinOp {
    Add,
    Mul,
    Sub,
    Div,
}

public class StreamingBinOp(
    private var prevLhs: StreamTensor,
    private var prevRhs: StreamTensor,
    public val op: BinOp,
    public val dim: D,
) {
    public companion object {
        public fun new(
            op: BinOp,
            dim: D,
        ): StreamingBinOp = StreamingBinOp(StreamTensor.empty(), StreamTensor.empty(), op, dim)
    }

    public fun resetState() {
        prevLhs.reset()
        prevRhs.reset()
    }

    public fun forward(
        lhs: Tensor,
        rhs: Tensor,
    ): Tensor =
        when (op) {
            BinOp.Add -> lhs.add(rhs)
            BinOp.Mul -> lhs.mul(rhs)
            BinOp.Sub -> lhs.sub(rhs)
            BinOp.Div -> lhs.div(rhs)
        }

    public fun step(
        lhs: StreamTensor,
        rhs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor {
        val catLhs = StreamTensor.cat2(prevLhs, lhs, dim)
        val catRhs = StreamTensor.cat2(prevRhs, rhs, dim)
        val lhsLen = catLhs.seqLen(dim)
        val rhsLen = catRhs.seqLen(dim)
        val commonLen = min(lhsLen, rhsLen)
        val lhsSplit = catLhs.split(dim, commonLen)
        val rhsSplit = catRhs.split(dim, commonLen)
        val lhsTensor = lhsSplit.lhs.asOption()
        val rhsTensor = rhsSplit.lhs.asOption()
        val ys =
            if (lhsTensor != null && rhsTensor != null) {
                StreamTensor.fromTensor(forward(lhsTensor, rhsTensor))
            } else if (lhsTensor == null && rhsTensor == null) {
                StreamTensor.empty()
            } else {
                bail("INTERNAL ERROR inconsistent lhs and rhs ${lhsSplit.lhs} ${rhsSplit.lhs}")
            }
        if (!mask.isEmpty() && (!lhsSplit.rhs.isEmpty() || !rhsSplit.rhs.isEmpty())) {
            bail(
                "cannot use a stream mask with a streaming bin op ${lhsSplit.rhs} ${rhsSplit.rhs} ${lhsSplit.lhs} ${rhsSplit.lhs}",
            )
        }
        prevLhs = lhsSplit.rhs
        prevRhs = rhsSplit.rhs
        return ys
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        val lhsVal = prevLhs.asOption()
        if (lhsVal != null) {
            val v = lhsVal.contiguous()
            v.narrow(0, batchIdx, 1).zeroSet()
            prevLhs = StreamTensor.fromTensor(v)
        }
        val rhsVal = prevRhs.asOption()
        if (rhsVal != null) {
            val v = rhsVal.contiguous()
            v.narrow(0, batchIdx, 1).zeroSet()
            prevRhs = StreamTensor.fromTensor(v)
        }
    }
}

/**
 * Simple wrapper that doesn't do any buffering.
 */
public class Map(
    private val module: Module,
) : StreamingModule {
    override fun resetState() {}

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor = xs.apply(module)
}
