// port-lint: source seanet.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candlenn.Activation
import io.github.kotlinmania.candlenn.VarBuilder
import kotlin.math.pow

public data class SeaNetConfig(
    public val dimension: Int,
    public val channels: Int,
    public val causal: Boolean,
    public val nFilters: Int,
    public val nResidualLayers: Int,
    public val ratios: List<Int>,
    public val activation: Activation,
    public val norm: Norm,
    public val kernelSize: Int,
    public val residualKernelSize: Int,
    public val lastKernelSize: Int,
    public val dilationBase: Int,
    public val padMode: PadMode,
    public val trueSkip: Boolean,
    public val compress: Int,
    public val lstm: Int,
    public val disableNormOuterBlocks: Int,
    public val finalActivation: Activation?,
)

public data class KernelSizeAndDilation(
    public val kernelSize: Int,
    public val dilation: Int,
)

public class SeaNetResnetBlock(
    private val block: List<StreamableConv1d>,
    private val shortcut: StreamableConv1d?,
    private val activation: Activation,
    private val skipOp: StreamingBinOp,
) : Module,
    StreamingModule {
    public companion object {
        public fun new(
            dim: Int,
            kSizesAndDilations: List<KernelSizeAndDilation>,
            activation: Activation,
            norm: Norm?,
            causal: Boolean,
            padMode: PadMode,
            compress: Int,
            trueSkip: Boolean,
            vb: VarBuilder,
        ): SeaNetResnetBlock {
            val block = ArrayList<StreamableConv1d>(kSizesAndDilations.size)
            val hidden = dim / compress
            val vbB = vb.pp("block")
            for ((i, kd) in kSizesAndDilations.withIndex()) {
                val inC = if (i == 0) dim else hidden
                val outC = if (i == kSizesAndDilations.size - 1) dim else hidden
                val c =
                    StreamableConv1d.new(
                        inC = inC,
                        outC = outC,
                        kSize = kd.kernelSize,
                        stride = 1,
                        dilation = kd.dilation,
                        groups = 1,
                        bias = true,
                        causal = causal,
                        norm = norm,
                        padMode = padMode,
                        vb = vbB.pp(2 * i + 1),
                    )
                block.add(c)
            }
            val shortcut =
                if (trueSkip) {
                    null
                } else {
                    StreamableConv1d.new(
                        inC = dim,
                        outC = dim,
                        kSize = 1,
                        stride = 1,
                        dilation = 1,
                        groups = 1,
                        bias = true,
                        causal = causal,
                        norm = norm,
                        padMode = padMode,
                        vb = vb.pp("shortcut"),
                    )
                }
            return SeaNetResnetBlock(
                block = block,
                shortcut = shortcut,
                activation = activation,
                skipOp = StreamingBinOp.new(BinOp.Add, D.Minus1),
            )
        }
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        for (b in block) {
            b.resetBatchIdx(batchIdx, batchSize)
        }
        shortcut?.resetBatchIdx(batchIdx, batchSize)
        skipOp.resetBatchIdx(batchIdx, batchSize)
    }

    override fun forward(xs: Tensor): Tensor {
        var ys = xs
        for (b in block) {
            ys = ys.apply(activation).apply(b)
        }
        return if (shortcut == null) {
            ys.add(xs)
        } else {
            ys.add(xs.apply(shortcut))
        }
    }

    override fun resetState() {
        skipOp.resetState()
        for (b in block) {
            b.resetState()
        }
        shortcut?.resetState()
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor {
        var ys = xs
        for (b in block) {
            ys = b.step(ys.apply(activation), mask)
        }
        return if (shortcut == null) {
            skipOp.step(ys, xs, mask)
        } else {
            skipOp.step(ys, shortcut.step(xs, mask), mask)
        }
    }
}

internal class EncoderLayer(
    val residuals: List<SeaNetResnetBlock>,
    val downsample: StreamableConv1d,
)

public class SeaNetEncoder private constructor(
    private val initConv1d: StreamableConv1d,
    private val activation: Activation,
    private val layers: List<EncoderLayer>,
    private val finalConv1d: StreamableConv1d,
) : Module,
    StreamingModule {
    public companion object {
        public fun new(
            cfg: SeaNetConfig,
            vb: VarBuilder,
        ): SeaNetEncoder {
            if (cfg.lstm > 0) {
                bail("seanet lstm is not supported")
            }
            val nBlocks = 2 + cfg.ratios.size
            var mult = 1
            val initNorm = if (cfg.disableNormOuterBlocks >= 1) null else cfg.norm
            var layerIdx = 0
            val vbModel = vb.pp("model")
            val initConv1d =
                StreamableConv1d.new(
                    inC = cfg.channels,
                    outC = mult * cfg.nFilters,
                    kSize = cfg.kernelSize,
                    stride = 1,
                    dilation = 1,
                    groups = 1,
                    bias = true,
                    causal = cfg.causal,
                    norm = initNorm,
                    padMode = cfg.padMode,
                    vb = vbModel.pp(layerIdx),
                )
            layerIdx += 1
            val layers = ArrayList<EncoderLayer>(cfg.ratios.size)

            for ((i, ratio) in cfg.ratios.asReversed().withIndex()) {
                val norm = if (cfg.disableNormOuterBlocks >= i + 2) null else cfg.norm
                val residuals = ArrayList<SeaNetResnetBlock>(cfg.nResidualLayers)
                for (j in 0 until cfg.nResidualLayers) {
                    val dilation =
                        cfg.dilationBase
                            .toDouble()
                            .pow(j)
                            .toInt()
                    val resnetBlock =
                        SeaNetResnetBlock.new(
                            dim = mult * cfg.nFilters,
                            kSizesAndDilations =
                                listOf(
                                    KernelSizeAndDilation(cfg.residualKernelSize, dilation),
                                    KernelSizeAndDilation(1, 1),
                                ),
                            activation = cfg.activation,
                            norm = norm,
                            causal = cfg.causal,
                            padMode = cfg.padMode,
                            compress = cfg.compress,
                            trueSkip = cfg.trueSkip,
                            vb = vbModel.pp(layerIdx),
                        )
                    residuals.add(resnetBlock)
                    layerIdx += 1
                }
                val downsample =
                    StreamableConv1d.new(
                        inC = mult * cfg.nFilters,
                        outC = mult * cfg.nFilters * 2,
                        kSize = ratio * 2,
                        stride = ratio,
                        dilation = 1,
                        groups = 1,
                        bias = true,
                        causal = true,
                        norm = norm,
                        padMode = cfg.padMode,
                        vb = vbModel.pp(layerIdx + 1),
                    )
                layerIdx += 2
                layers.add(EncoderLayer(residuals = residuals, downsample = downsample))
                mult *= 2
            }

            val finalNorm = if (cfg.disableNormOuterBlocks >= nBlocks) null else cfg.norm
            val finalConv1d =
                StreamableConv1d.new(
                    inC = mult * cfg.nFilters,
                    outC = cfg.dimension,
                    kSize = cfg.lastKernelSize,
                    stride = 1,
                    dilation = 1,
                    groups = 1,
                    bias = true,
                    causal = cfg.causal,
                    norm = finalNorm,
                    padMode = cfg.padMode,
                    vb = vbModel.pp(layerIdx + 1),
                )
            return SeaNetEncoder(
                initConv1d = initConv1d,
                activation = cfg.activation,
                layers = layers,
                finalConv1d = finalConv1d,
            )
        }
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        initConv1d.resetBatchIdx(batchIdx, batchSize)
        finalConv1d.resetBatchIdx(batchIdx, batchSize)
        for (layer in layers) {
            layer.downsample.resetBatchIdx(batchIdx, batchSize)
            for (l in layer.residuals) {
                l.resetBatchIdx(batchIdx, batchSize)
            }
        }
    }

    override fun forward(xs: Tensor): Tensor {
        var cur = xs.apply(initConv1d)
        for (layer in layers) {
            for (residual in layer.residuals) {
                cur = cur.apply(residual)
            }
            cur = cur.apply(activation).apply(layer.downsample)
        }
        return cur.apply(activation).apply(finalConv1d)
    }

    override fun resetState() {
        initConv1d.resetState()
        for (layer in layers) {
            for (residual in layer.residuals) {
                residual.resetState()
            }
            layer.downsample.resetState()
        }
        finalConv1d.resetState()
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor {
        var cur = initConv1d.step(xs, mask)
        for (layer in layers) {
            for (residual in layer.residuals) {
                cur = residual.step(cur, mask)
            }
            cur = layer.downsample.step(cur.apply(activation), mask)
        }
        return finalConv1d.step(cur.apply(activation), mask)
    }
}

internal class DecoderLayer(
    val upsample: StreamableConvTranspose1d,
    val residuals: List<SeaNetResnetBlock>,
)

public class SeaNetDecoder private constructor(
    private val initConv1d: StreamableConv1d,
    private val activation: Activation,
    private val layers: List<DecoderLayer>,
    private val finalConv1d: StreamableConv1d,
    private val finalActivation: Activation?,
) : Module,
    StreamingModule {
    public companion object {
        public fun new(
            cfg: SeaNetConfig,
            vb: VarBuilder,
        ): SeaNetDecoder {
            if (cfg.lstm > 0) {
                bail("seanet lstm is not supported")
            }
            val nBlocks = 2 + cfg.ratios.size
            var mult = 1 shl cfg.ratios.size
            val initNorm = if (cfg.disableNormOuterBlocks == nBlocks) null else cfg.norm
            var layerIdx = 0
            val vbModel = vb.pp("model")
            val initConv1d =
                StreamableConv1d.new(
                    inC = cfg.dimension,
                    outC = mult * cfg.nFilters,
                    kSize = cfg.kernelSize,
                    stride = 1,
                    dilation = 1,
                    groups = 1,
                    bias = true,
                    causal = cfg.causal,
                    norm = initNorm,
                    padMode = cfg.padMode,
                    vb = vbModel.pp(layerIdx),
                )
            layerIdx += 1
            val layers = ArrayList<DecoderLayer>(cfg.ratios.size)
            for ((i, ratio) in cfg.ratios.withIndex()) {
                val norm = if (cfg.disableNormOuterBlocks + i + 1 >= nBlocks) null else cfg.norm
                val upsample =
                    StreamableConvTranspose1d.new(
                        inC = mult * cfg.nFilters,
                        outC = mult * cfg.nFilters / 2,
                        kSize = ratio * 2,
                        stride = ratio,
                        groups = 1,
                        bias = true,
                        causal = true,
                        norm = norm,
                        vb = vbModel.pp(layerIdx + 1),
                    )
                layerIdx += 2

                val residuals = ArrayList<SeaNetResnetBlock>(cfg.nResidualLayers)
                for (j in 0 until cfg.nResidualLayers) {
                    val dilation =
                        cfg.dilationBase
                            .toDouble()
                            .pow(j)
                            .toInt()
                    val resnetBlock =
                        SeaNetResnetBlock.new(
                            dim = mult * cfg.nFilters / 2,
                            kSizesAndDilations =
                                listOf(
                                    KernelSizeAndDilation(cfg.residualKernelSize, dilation),
                                    KernelSizeAndDilation(1, 1),
                                ),
                            activation = cfg.activation,
                            norm = norm,
                            causal = cfg.causal,
                            padMode = cfg.padMode,
                            compress = cfg.compress,
                            trueSkip = cfg.trueSkip,
                            vb = vbModel.pp(layerIdx),
                        )
                    residuals.add(resnetBlock)
                    layerIdx += 1
                }
                layers.add(DecoderLayer(upsample = upsample, residuals = residuals))
                mult /= 2
            }
            val finalNorm = if (cfg.disableNormOuterBlocks >= 1) null else cfg.norm
            val finalConv1d =
                StreamableConv1d.new(
                    inC = cfg.nFilters,
                    outC = cfg.channels,
                    kSize = cfg.lastKernelSize,
                    stride = 1,
                    dilation = 1,
                    groups = 1,
                    bias = true,
                    causal = cfg.causal,
                    norm = finalNorm,
                    padMode = cfg.padMode,
                    vb = vbModel.pp(layerIdx + 1),
                )
            return SeaNetDecoder(
                initConv1d = initConv1d,
                activation = cfg.activation,
                layers = layers,
                finalConv1d = finalConv1d,
                finalActivation = cfg.finalActivation,
            )
        }
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        initConv1d.resetBatchIdx(batchIdx, batchSize)
        finalConv1d.resetBatchIdx(batchIdx, batchSize)
        for (layer in layers) {
            layer.upsample.resetBatchIdx(batchIdx, batchSize)
            for (l in layer.residuals) {
                l.resetBatchIdx(batchIdx, batchSize)
            }
        }
    }

    override fun forward(xs: Tensor): Tensor {
        var cur = xs.apply(initConv1d)
        for (layer in layers) {
            cur = cur.apply(activation).apply(layer.upsample)
            for (residual in layer.residuals) {
                cur = cur.apply(residual)
            }
        }
        cur = cur.apply(activation).apply(finalConv1d)
        return if (finalActivation == null) {
            cur
        } else {
            cur.apply(finalActivation)
        }
    }

    override fun resetState() {
        initConv1d.resetState()
        for (layer in layers) {
            for (residual in layer.residuals) {
                residual.resetState()
            }
            layer.upsample.resetState()
        }
        finalConv1d.resetState()
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor {
        var cur = initConv1d.step(xs, mask)
        for (layer in layers) {
            cur = layer.upsample.step(cur.apply(activation), mask)
            for (residual in layer.residuals) {
                cur = residual.step(cur, mask)
            }
        }
        cur = finalConv1d.step(cur.apply(activation), mask)
        return cur.apply(finalActivation)
    }
}
