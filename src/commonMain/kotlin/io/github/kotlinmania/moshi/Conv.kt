// port-lint: source conv.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.CudnnFwdAlgo
import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candle.convTranspose1d
import io.github.kotlinmania.candlenn.Conv1d
import io.github.kotlinmania.candlenn.Conv1dConfig
import io.github.kotlinmania.candlenn.GroupNorm
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candlenn.conv1d
import io.github.kotlinmania.candlenn.conv1dNoBias
import io.github.kotlinmania.candlenn.groupNorm
import kotlin.math.ceil
import kotlin.math.max

public enum class Norm {
    WeightNorm,
    SpectralNorm,
    TimeGroupNorm,
}

public enum class PadMode {
    Constant,
    Reflect,
    Replicate,
}

/**
 * Applies weight norm for inference by recomputing the weight tensor. This
 * does not apply to training.
 */
private fun conv1dWeightNorm(
    inC: Int,
    outC: Int,
    kernelSize: Int,
    bias: Boolean,
    config: Conv1dConfig,
    vb: VarBuilder,
): Conv1d {
    val weight =
        if (vb.containsTensor("weight")) {
            vb.get(Shape(outC, inC, kernelSize), "weight")
        } else {
            val weightG = vb.get(Shape(outC, 1, 1), "weight_g")
            val weightV = vb.get(Shape(outC, inC, kernelSize), "weight_v")
            val normV = weightV.sqr().sumKeepdim(listOf(1, 2)).sqrt()
            weightV.broadcastMul(weightG).broadcastDiv(normV)
        }
    val biasTensor = if (bias) vb.get(outC, "bias") else null
    return Conv1d.new(weight, biasTensor, config)
}

public class NormConv1d(
    internal val conv: Conv1d,
    internal val norm: GroupNorm?,
) : Module {
    public companion object {
        public fun new(
            inC: Int,
            outC: Int,
            kSize: Int,
            causal: Boolean,
            norm: Norm?,
            bias: Boolean,
            cfg: Conv1dConfig,
            vb: VarBuilder,
        ): NormConv1d {
            val conv =
                when (norm) {
                    null, Norm.TimeGroupNorm -> {
                        if (bias) {
                            conv1d(inC, outC, kSize, cfg, vb.pp("conv"))
                        } else {
                            conv1dNoBias(inC, outC, kSize, cfg, vb.pp("conv"))
                        }
                    }

                    Norm.WeightNorm -> {
                        conv1dWeightNorm(inC, outC, kSize, bias, cfg, vb.pp("conv"))
                    }

                    Norm.SpectralNorm -> {
                        bail("SpectralNorm is not supported yet.")
                    }
                }
            val groupNormLayer =
                when (norm) {
                    null, Norm.WeightNorm, Norm.SpectralNorm -> {
                        null
                    }

                    Norm.TimeGroupNorm -> {
                        if (causal) {
                            bail("GroupNorm doesn't support causal evaluation.")
                        }
                        groupNorm(1, outC, 1e-5, vb.pp("norm"))
                    }
                }
            return NormConv1d(conv, groupNormLayer)
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val convOut = xs.apply(conv)
        return if (norm == null) {
            convOut
        } else {
            convOut.apply(norm)
        }
    }
}

public class NormConvTranspose1d(
    internal val ws: Tensor,
    internal val bs: Tensor?,
    internal val kSize: Int,
    internal val stride: Int,
    internal val groups: Int,
    internal val norm: GroupNorm?,
) : Module {
    public companion object {
        public fun new(
            inC: Int,
            outC: Int,
            kSize: Int,
            causal: Boolean,
            norm: Norm?,
            bias: Boolean,
            stride: Int,
            groups: Int,
            vb: VarBuilder,
        ): NormConvTranspose1d {
            val vbConvTr = vb.pp("convtr")
            val bs = if (bias) vbConvTr.get(outC, "bias") else null
            val rawWs =
                when (norm) {
                    null, Norm.TimeGroupNorm -> {
                        vbConvTr.get(Shape(inC, outC / groups, kSize), "weight")
                    }

                    Norm.WeightNorm -> {
                        if (vbConvTr.containsTensor("weight")) {
                            vbConvTr.get(Shape(inC, outC, kSize), "weight")
                        } else {
                            val weightG = vbConvTr.get(Shape(inC, 1, 1), "weight_g")
                            val weightV = vbConvTr.get(Shape(inC, outC, kSize), "weight_v")
                            val normV = weightV.sqr().sumKeepdim(listOf(1, 2)).sqrt()
                            weightV.broadcastMul(weightG).broadcastDiv(normV)
                        }
                    }

                    Norm.SpectralNorm -> {
                        bail("SpectralNorm is not supported yet.")
                    }
                }
            val (ws, resolvedGroups) =
                if (groups == outC && inC == outC) {
                    val eye = Tensor.eye(outC, rawWs.dtype(), rawWs.device())
                    val expandedWs =
                        rawWs
                            .repeat(Shape(1, outC, 1))
                            .mul(eye.unsqueeze(2).repeat(Shape(1, 1, kSize)))
                    expandedWs to 1
                } else {
                    rawWs to groups
                }
            val groupNormLayer =
                when (norm) {
                    null, Norm.WeightNorm, Norm.SpectralNorm -> {
                        null
                    }

                    Norm.TimeGroupNorm -> {
                        if (causal) {
                            bail("GroupNorm doesn't support causal evaluation.")
                        }
                        groupNorm(1, outC, 1e-5, vbConvTr.pp("norm"))
                    }
                }
            return NormConvTranspose1d(
                ws = ws,
                bs = bs,
                kSize = kSize,
                stride = stride,
                groups = resolvedGroups,
                norm = groupNormLayer,
            )
        }
    }

    override fun forward(xs: Tensor): Tensor {
        var out = xs.convTranspose1d(ws, 0, 0, stride, 1, groups)
        if (bs != null) {
            val b = bs.dims1()
            val biasReshaped = bs.reshape(Shape(1, b, 1))
            out = out.broadcastAdd(biasReshaped)
        }
        return if (norm == null) {
            out
        } else {
            out.apply(norm)
        }
    }
}

private fun getExtraPaddingForConv1d(
    xs: Tensor,
    kSize: Int,
    stride: Int,
    paddingTotal: Int,
): Int {
    val len = xs.dim(D.Minus1)
    val nFrames = max(0, len + paddingTotal - kSize).toDouble() / stride.toDouble() + 1.0
    val idealLen = max(0, (ceil(nFrames).toInt() - 1) * stride + kSize - paddingTotal)
    return max(0, idealLen - len)
}

private fun pad1d(
    xs: Tensor,
    padL: Int,
    padR: Int,
    mode: PadMode,
): Tensor =
    when (mode) {
        PadMode.Constant -> xs.padWithZeros(D.Minus1, padL, padR)
        PadMode.Reflect -> bail("pad-mode 'reflect' is not supported")
        PadMode.Replicate -> xs.padWithSame(D.Minus1, padL, padR)
    }

private fun unpad1d(
    xs: Tensor,
    unpadL: Int,
    unpadR: Int,
): Tensor {
    val len = xs.dim(D.Minus1)
    if (len < unpadL + unpadR) {
        bail("unpad1d: tensor len $len is too low, $unpadL + $unpadR")
    }
    return xs.narrow(D.Minus1, unpadL, len - (unpadL + unpadR))
}

public class StreamableConv1d(
    private val conv: NormConv1d,
    private val causal: Boolean,
    private val padMode: PadMode,
    private var statePrevXs: StreamTensor,
    private var leftPadApplied: Boolean,
    private val kernelSize: Int,
) : Module,
    StreamingModule {
    public companion object {
        public fun new(
            inC: Int,
            outC: Int,
            kSize: Int,
            stride: Int,
            dilation: Int,
            groups: Int,
            bias: Boolean,
            causal: Boolean,
            norm: Norm?,
            padMode: PadMode,
            vb: VarBuilder,
        ): StreamableConv1d {
            val cfg =
                Conv1dConfig(
                    padding = 0,
                    stride = stride,
                    dilation = dilation,
                    groups = groups,
                    cudnnFwdAlgo = CudnnFwdAlgo.ImplicitGemm,
                )
            val conv = NormConv1d.new(inC, outC, kSize, causal, norm, bias, cfg, vb.pp("conv"))
            if (kSize < stride) {
                bail("kernel-size $kSize is smaller than stride $stride")
            }
            return StreamableConv1d(
                conv = conv,
                causal = causal,
                padMode = padMode,
                statePrevXs = StreamTensor.empty(),
                leftPadApplied = false,
                kernelSize = kSize,
            )
        }
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        val v = statePrevXs.asOption()
        if (v != null) {
            val contiguous = v.contiguous()
            contiguous.narrow(0, batchIdx, 1).zeroSet()
            statePrevXs = StreamTensor.fromTensor(contiguous)
        }
    }

    override fun forward(xs: Tensor): Tensor {
        xs.dims3()
        val weightKSize = conv.conv.weight().dim(D.Minus1)
        val convCfg = conv.conv.config()
        val kSize = (weightKSize - 1) * convCfg.dilation + 1
        val paddingTotal = kSize - convCfg.stride
        val extraPadding = getExtraPaddingForConv1d(xs, kSize, convCfg.stride, paddingTotal)
        val padded =
            if (causal) {
                pad1d(xs, paddingTotal, extraPadding, padMode)
            } else {
                val paddingRight = paddingTotal / 2
                val paddingLeft = paddingTotal - paddingRight
                pad1d(xs, paddingLeft, paddingRight + extraPadding, padMode)
            }
        return padded.apply(conv)
    }

    override fun resetState() {
        statePrevXs.reset()
        leftPadApplied = false
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor {
        val rawXs = xs.asOption() ?: return StreamTensor.empty()
        val paddedXs =
            if (leftPadApplied) {
                rawXs
            } else {
                leftPadApplied = true
                val weightKSize = conv.conv.weight().dim(D.Minus1)
                val convCfg = conv.conv.config()
                val kSize = (weightKSize - 1) * convCfg.dilation + 1
                val paddingTotal = kSize - convCfg.stride
                pad1d(rawXs, paddingTotal, 0, padMode)
            }
        val cfg = conv.conv.config()
        val stride = cfg.stride
        val dilation = cfg.dilation
        val kernel = (kernelSize - 1) * dilation + 1
        val catXs = StreamTensor.cat2(statePrevXs, StreamTensor.fromTensor(paddedXs), D.Minus1)
        val seqLen = catXs.seqLen(D.Minus1)
        val numFrames = max(0, seqLen + stride - kernel) / stride
        val (nextStatePrevXs, ys) =
            if (numFrames > 0) {
                val offset = numFrames * stride
                val nextPrev = catXs.narrow(D.Minus1, offset, seqLen - offset)
                val inL = (numFrames - 1) * stride + kernel
                val inXs = catXs.narrow(D.Minus1, 0, inL)
                // We apply the underlying conv directly rather than through forward so as
                // not to apply any padding here.
                val outYs = inXs.apply(conv.conv)
                nextPrev to outYs
            } else {
                catXs to StreamTensor.empty()
            }
        val maskTensor = mask.asOption()
        val maskedStatePrevXs =
            if (maskTensor == null) {
                nextStatePrevXs
            } else {
                val prevOpt = nextStatePrevXs.asOption()
                val prevPrevOpt = statePrevXs.asOption()
                if (prevOpt == null && prevPrevOpt == null) {
                    nextStatePrevXs
                } else if (prevOpt != null && prevPrevOpt == null) {
                    val z = prevOpt.zerosLike()
                    val m =
                        maskTensor
                            .reshape(Shape(maskTensor.elemCount(), 1, 1))
                            .broadcastAs(prevOpt.shape())
                    StreamTensor.fromTensor(m.whereCond(prevOpt, z))
                } else if (prevOpt == null && prevPrevOpt != null) {
                    bail("streaming conv1d should only be used with constant steps")
                } else {
                    val p = prevOpt!!
                    val pp = prevPrevOpt!!
                    if (p.shape() != pp.shape()) {
                        bail("streaming conv1d should only be used with constant steps $p $pp")
                    }
                    val m =
                        maskTensor
                            .reshape(Shape(maskTensor.elemCount(), 1, 1))
                            .broadcastAs(p.shape())
                    StreamTensor.fromTensor(m.whereCond(p, pp))
                }
            }
        statePrevXs = maskedStatePrevXs
        return ys
    }
}

public class StreamableConvTranspose1d(
    private val convtr: NormConvTranspose1d,
    private val causal: Boolean,
    private var statePrevYs: StreamTensor,
    private val kernelSize: Int,
) : Module,
    StreamingModule {
    public companion object {
        public fun new(
            inC: Int,
            outC: Int,
            kSize: Int,
            stride: Int,
            groups: Int,
            bias: Boolean,
            causal: Boolean,
            norm: Norm?,
            vb: VarBuilder,
        ): StreamableConvTranspose1d {
            val convtr =
                NormConvTranspose1d.new(
                    inC = inC,
                    outC = outC,
                    kSize = kSize,
                    causal = causal,
                    norm = norm,
                    bias = bias,
                    stride = stride,
                    groups = groups,
                    vb = vb.pp("convtr"),
                )
            return StreamableConvTranspose1d(
                convtr = convtr,
                causal = causal,
                statePrevYs = StreamTensor.empty(),
                kernelSize = kSize,
            )
        }
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        val v = statePrevYs.asOption()
        if (v != null) {
            val contiguous = v.contiguous()
            contiguous.narrow(0, batchIdx, 1).zeroSet()
            statePrevYs = StreamTensor.fromTensor(contiguous)
        }
    }

    override fun forward(xs: Tensor): Tensor {
        val kSize = convtr.kSize
        val stride = convtr.stride
        val paddingTotal = max(0, kSize - stride)
        val out = xs.apply(convtr)
        return if (causal) {
            // This corresponds to trimRightRatio = 1.
            unpad1d(out, 0, paddingTotal)
        } else {
            val paddingRight = paddingTotal / 2
            val paddingLeft = paddingTotal - paddingRight
            unpad1d(out, paddingLeft, paddingRight)
        }
    }

    override fun resetState() {
        statePrevYs.reset()
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor {
        val rawXs = xs.asOption() ?: return StreamTensor.empty()
        val stride = convtr.stride
        // We apply the underlying convtr directly rather than through forward so as
        // not to apply any padding here.
        val rawYs = convtr.forward(rawXs)
        val ot = rawYs.dim(D.Minus1)
        val prevYsOpt = statePrevYs.asOption()
        val combinedYs =
            if (prevYsOpt == null) {
                rawYs
            } else {
                val pt = prevYsOpt.dim(D.Minus1)
                // Remove the bias as it will be applied multiple times.
                val unbiasedPrevYs =
                    if (convtr.bs == null) {
                        prevYsOpt
                    } else {
                        val bias = convtr.bs.reshape(Shape(1, convtr.bs.elemCount(), 1))
                        prevYsOpt.broadcastSub(bias)
                    }
                val ys1 = rawYs.narrow(D.Minus1, 0, pt).add(unbiasedPrevYs)
                val ys2 = rawYs.narrow(D.Minus1, pt, ot - pt)
                Tensor.cat(listOf(ys1, ys2), D.Minus1)
            }
        val invalidSteps = kernelSize - stride
        val split = StreamTensor.fromTensor(combinedYs).split(D.Minus1, ot - invalidSteps)
        val ys = split.lhs
        val nextPrevYs = split.rhs
        val maskTensor = mask.asOption()
        val maskedPrevYs =
            if (maskTensor == null) {
                nextPrevYs
            } else {
                val prevOpt = nextPrevYs.asOption()
                val prevPrevOpt = statePrevYs.asOption()
                if (prevOpt == null && prevPrevOpt == null) {
                    nextPrevYs
                } else if (prevOpt != null && prevPrevOpt == null) {
                    val z = prevOpt.zerosLike()
                    val m =
                        maskTensor
                            .reshape(Shape(maskTensor.elemCount(), 1, 1))
                            .broadcastAs(prevOpt.shape())
                    StreamTensor.fromTensor(m.whereCond(prevOpt, z))
                } else if (prevOpt == null && prevPrevOpt != null) {
                    bail("streaming conv-tr1d should only be used with constant steps")
                } else {
                    val p = prevOpt!!
                    val pp = prevPrevOpt!!
                    if (p.shape() != pp.shape()) {
                        bail("streaming conv-tr1d should only be used with constant steps $p $pp")
                    }
                    val m =
                        maskTensor
                            .reshape(Shape(maskTensor.elemCount(), 1, 1))
                            .broadcastAs(p.shape())
                    StreamTensor.fromTensor(m.whereCond(p, pp))
                }
            }
        statePrevYs = maskedPrevYs
        return ys
    }
}

public class ConvDownsample1d(
    private val conv: StreamableConv1d,
) : Module,
    StreamingModule {
    public companion object {
        public fun new(
            stride: Int,
            dim: Int,
            causal: Boolean,
            learnt: Boolean,
            vb: VarBuilder,
        ): ConvDownsample1d {
            if (!learnt) {
                bail("only learnt=true is supported")
            }
            val conv =
                StreamableConv1d.new(
                    inC = dim,
                    outC = dim,
                    kSize = 2 * stride,
                    stride = stride,
                    dilation = 1,
                    groups = 1,
                    bias = false,
                    causal = causal,
                    norm = null,
                    padMode = PadMode.Replicate,
                    vb = vb.pp("conv"),
                )
            return ConvDownsample1d(conv)
        }
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        conv.resetBatchIdx(batchIdx, batchSize)
    }

    override fun forward(xs: Tensor): Tensor = xs.apply(conv)

    override fun resetState() {
        conv.resetState()
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor = conv.step(xs, mask)
}

public class ConvTrUpsample1d(
    private val convtr: StreamableConvTranspose1d,
) : Module,
    StreamingModule {
    public companion object {
        public fun new(
            stride: Int,
            dim: Int,
            causal: Boolean,
            learnt: Boolean,
            vb: VarBuilder,
        ): ConvTrUpsample1d {
            if (!learnt) {
                bail("only learnt=true is supported")
            }
            val convtr =
                StreamableConvTranspose1d.new(
                    inC = dim,
                    outC = dim,
                    kSize = 2 * stride,
                    stride = stride,
                    groups = dim,
                    bias = false,
                    causal = causal,
                    norm = null,
                    vb = vb.pp("convtr"),
                )
            return ConvTrUpsample1d(convtr)
        }
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        convtr.resetBatchIdx(batchIdx, batchSize)
    }

    override fun forward(xs: Tensor): Tensor = xs.apply(convtr)

    override fun resetState() {
        convtr.resetState()
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor = convtr.step(xs, mask)
}
