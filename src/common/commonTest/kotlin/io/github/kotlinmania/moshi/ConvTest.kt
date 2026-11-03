// port-lint: tests conv.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.candlenn.VarBuilder
import io.github.kotlinmania.candlenn.VarMap
import kotlin.test.Test

private fun runConv1d(
    kSize: Int,
    stride: Int,
    dilation: Int,
    stepSize: Int,
    len: Int,
    bias: Boolean,
) {
    val dev = Device.Cpu
    val vm = VarMap.new()
    val vb = VarBuilder.fromVarmap(vm, DType.F32, dev)
    val conv1d =
        StreamableConv1d.new(
            inC = 2,
            outC = 3,
            kSize = kSize,
            stride = stride,
            dilation = dilation,
            groups = 1,
            bias = bias,
            causal = true,
            norm = null,
            padMode = PadMode.Constant,
            vb = vb,
        )
    val xs = Tensor.randn(0f, 1f, Shape(1, 2, stepSize * len), dev)
    val ys = conv1d.forward(xs)
    val ysSteps = mutableListOf<Tensor>()
    for (idx in 0 until len) {
        val xsSlice = xs.narrow(2, stepSize * idx, stepSize)
        val ysStep = conv1d.step(StreamTensor.fromTensor(xsSlice), StreamMask.empty())
        val ysOpt = ysStep.asOption()
        if (ysOpt != null) {
            ysSteps.add(ysOpt)
        }
    }
    val ysCat = Tensor.cat(ysSteps, D.Minus1)
    val diff =
        ys
            .sub(ysCat)
            .abs()
            .flattenAll()
            .max(0)
            .toVec0<Float>()
    if (diff > 1e-5f) {
        bail("larger diff than expected $diff")
    }
}

private fun runConvTr1d(
    kSize: Int,
    stride: Int,
    stepSize: Int,
    len: Int,
    bias: Boolean,
) {
    val dev = Device.Cpu
    val vm = VarMap.new()
    val vb = VarBuilder.fromVarmap(vm, DType.F32, dev)
    val conv1d =
        StreamableConvTranspose1d.new(
            inC = 2,
            outC = 3,
            kSize = kSize,
            stride = stride,
            groups = 1,
            bias = bias,
            causal = true,
            norm = null,
            vb = vb,
        )
    val xs = Tensor.randn(0f, 1f, Shape(1, 2, stepSize * len), dev)
    val ys = conv1d.forward(xs)
    val ysSteps = mutableListOf<Tensor>()
    for (idx in 0 until len) {
        val xsSlice = xs.narrow(2, stepSize * idx, stepSize)
        val ysStep = conv1d.step(StreamTensor.fromTensor(xsSlice), StreamMask.empty())
        val ysOpt = ysStep.asOption()
        if (ysOpt != null) {
            ysSteps.add(ysOpt)
        }
    }
    val ysCat = Tensor.cat(ysSteps, D.Minus1)
    val diff =
        ys
            .sub(ysCat)
            .abs()
            .flattenAll()
            .max(0)
            .toVec0<Float>()
    if (diff > 1e-5f) {
        bail("larger diff than expected $diff")
    }
}

class ConvTest {
    @Test
    fun conv1d() {
        for (stepSize in listOf(1, 2, 3)) {
            for (bias in listOf(false, true)) {
                runConv1d(1, 1, 1, stepSize, 5, bias)
                runConv1d(2, 1, 1, stepSize, 5, bias)
                runConv1d(2, 2, 1, stepSize, 6, bias)
                runConv1d(3, 2, 1, stepSize, 8, bias)
                runConv1d(3, 2, 2, stepSize, 8, bias)
            }
        }
    }

    @Test
    fun convTr1d() {
        for (stepSize in listOf(1, 2, 3)) {
            for (bias in listOf(false, true)) {
                runConvTr1d(1, 1, stepSize, 5, bias)
                runConvTr1d(2, 1, stepSize, 5, bias)
                runConvTr1d(3, 1, stepSize, 5, bias)
                runConvTr1d(3, 2, stepSize, 5, bias)
            }
        }
    }
}
