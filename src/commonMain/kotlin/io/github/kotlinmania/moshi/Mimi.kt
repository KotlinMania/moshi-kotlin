// port-lint: source mimi.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candlenn.Activation
import io.github.kotlinmania.candlenn.VarBuilder

public enum class ResampleMethod {
    Conv,
    Interpolate,
}

public data class MimiConfig(
    public val channels: Int,
    public val sampleRate: Double,
    public val frameRate: Double,
    public val renormalize: Boolean,
    public val resampleMethod: ResampleMethod,
    public val seanet: SeaNetConfig,
    public val transformer: TransformerConfig,
    public val quantizerNQ: Int,
    public val quantizerBins: Int,
    public val quantizerDim: Int,
) {
    public companion object {
        public fun v01(numCodebooks: Int? = null): MimiConfig {
            val seanetCfg =
                SeaNetConfig(
                    dimension = 512,
                    channels = 1,
                    causal = true,
                    nFilters = 64,
                    nResidualLayers = 1,
                    activation = Activation.Elu(1.0),
                    compress = 2,
                    dilationBase = 2,
                    disableNormOuterBlocks = 0,
                    finalActivation = null,
                    kernelSize = 7,
                    residualKernelSize = 3,
                    lastKernelSize = 3,
                    lstm = 0,
                    norm = Norm.WeightNorm,
                    padMode = PadMode.Constant,
                    ratios = listOf(8, 6, 5, 4),
                    trueSkip = true,
                )
            val transformerCfg =
                TransformerConfig(
                    dModel = seanetCfg.dimension,
                    numHeads = 8,
                    numLayers = 8,
                    causal = true,
                    normFirst = true,
                    biasFf = false,
                    biasAttn = false,
                    layerScale = 0.01,
                    context = 250,
                    convKernelSize = 5,
                    useConvBias = true,
                    useConvBlock = false,
                    crossAttention = null,
                    maxPeriod = 10000,
                    gating = null,
                    norm = NormType.LayerNorm,
                    positionalEmbedding = PositionalEmbedding.Rope,
                    dimFeedforward = 2048,
                    kvRepeat = 1,
                    convLayout = true,
                    maxSeqLen = 8192,
                    sharedCrossAttn = false,
                )
            return MimiConfig(
                channels = 1,
                sampleRate = 24_000.0,
                frameRate = 12.5,
                renormalize = true,
                resampleMethod = ResampleMethod.Conv,
                seanet = seanetCfg,
                transformer = transformerCfg,
                quantizerNQ = numCodebooks ?: 16,
                quantizerBins = 2048,
                quantizerDim = 256,
            )
        }
    }
}

public class Mimi(
    private val encoder: SeaNetEncoder,
    private val decoder: SeaNetDecoder,
    private val encoderTransformer: Transformer,
    private val decoderTransformer: Transformer,
    private val downsample: ConvDownsample1d,
    private val upsample: ConvTrUpsample1d,
    private val quantizer: SplitResidualVectorQuantizer,
    private val config: MimiConfig,
) {
    public companion object {
        public fun new(
            cfg: MimiConfig,
            vb: VarBuilder,
        ): Mimi = newInternal(null, cfg, vb)

        public fun batched(
            batchSize: Int,
            cfg: MimiConfig,
            vb: VarBuilder,
        ): Mimi = newInternal(batchSize, cfg, vb)

        internal fun newInternal(
            batchSize: Int?,
            cfg: MimiConfig,
            vb: VarBuilder,
        ): Mimi {
            val dim = cfg.seanet.dimension
            val encoder = SeaNetEncoder.new(cfg.seanet, vb.pp("encoder"))
            val decoder = SeaNetDecoder.new(cfg.seanet, vb.pp("decoder"))
            val encoderTransformer =
                Transformer.new(
                    batchSize = batchSize,
                    dim = dim,
                    cfg = cfg.transformer,
                    vb = vb.pp("encoder_transformer"),
                )
            val decoderTransformer =
                Transformer.new(
                    batchSize = batchSize,
                    dim = dim,
                    cfg = cfg.transformer,
                    vb = vb.pp("decoder_transformer"),
                )
            val quantizer =
                SplitResidualVectorQuantizer.new(
                    dim = cfg.quantizerDim,
                    inputDim = dim,
                    outputDim = dim,
                    nQ = cfg.quantizerNQ,
                    bins = cfg.quantizerBins,
                    vb = vb.pp("quantizer"),
                )
            val ratioProduct = cfg.seanet.ratios.fold(1) { acc, r -> acc * r }
            val encoderFrameRate = cfg.sampleRate / ratioProduct.toDouble()
            val downsampleStride = (encoderFrameRate / cfg.frameRate).toInt()
            val downsample =
                ConvDownsample1d.new(
                    stride = downsampleStride,
                    dim = dim,
                    causal = true,
                    learnt = true,
                    vb = vb.pp("downsample"),
                )
            val upsample =
                ConvTrUpsample1d.new(
                    stride = downsampleStride,
                    dim = dim,
                    causal = true,
                    learnt = true,
                    vb = vb.pp("upsample"),
                )
            return Mimi(
                encoder = encoder,
                decoder = decoder,
                encoderTransformer = encoderTransformer,
                decoderTransformer = decoderTransformer,
                downsample = downsample,
                upsample = upsample,
                quantizer = quantizer,
                config = cfg,
            )
        }
    }

    public fun config(): MimiConfig = config

    public fun encodePreQuantize(xs: Tensor): Tensor {
        val encOut = encoder.forward(xs)
        encoderTransformer.resetState()
        val trOut = encoderTransformer.forward(encOut)
        return trOut[0].apply(downsample)
    }

    public fun encode(xs: Tensor): Tensor {
        val encOut = encoder.forward(xs)
        encoderTransformer.resetState()
        val trOut = encoderTransformer.forward(encOut)
        val downsampled = trOut[0].apply(downsample)
        return quantizer.encode(downsampled)
    }

    public fun encodeStep(
        xs: StreamTensor,
        m: StreamMask,
    ): StreamTensor {
        val encOut = encoder.step(xs, m)
        val trOut = encoderTransformer.step(encOut, m)
        val downsampled = downsample.step(trOut, m)
        val raw = downsampled.asOption() ?: return StreamTensor.empty()
        val codes = quantizer.encode(raw)
        return StreamTensor.fromTensor(codes)
    }

    public fun decode(codes: Tensor): Tensor {
        val emb = quantizer.decode(codes)
        val upsampled = emb.apply(upsample)
        decoderTransformer.resetState()
        val outs = decoderTransformer.forward(upsampled)
        return decoder.forward(outs[0])
    }

    public fun decodeStep(
        codes: StreamTensor,
        m: StreamMask,
    ): StreamTensor {
        val rawCodes = codes.asOption()
        val emb =
            if (rawCodes != null) {
                StreamTensor.fromTensor(quantizer.decode(rawCodes))
            } else {
                StreamTensor.empty()
            }
        val upsampled = upsample.step(emb, m)
        val out = decoderTransformer.step(upsampled, m)
        return decoder.step(out, m)
    }

    public fun resetState() {
        encoder.resetState()
        encoderTransformer.resetState()
        decoder.resetState()
        decoderTransformer.resetState()
        upsample.resetState()
        downsample.resetState()
    }

    public fun resetBatchIdx(
        batchIdx: Int,
        batchSize: Int,
    ) {
        encoderTransformer.resetBatchIdx(batchIdx, batchSize)
        decoderTransformer.resetBatchIdx(batchIdx, batchSize)
        encoder.resetBatchIdx(batchIdx, batchSize)
        decoder.resetBatchIdx(batchIdx, batchSize)
        upsample.resetBatchIdx(batchIdx, batchSize)
        downsample.resetBatchIdx(batchIdx, batchSize)
    }
}

public fun loadMimi(
    modelFile: String,
    numCodebooks: Int?,
    dev: Device,
): Mimi {
    val vb = VarBuilder.fromMmapedSafetensors(listOf(modelFile), DType.F32, dev)
    val cfg = MimiConfig.v01(numCodebooks)
    return Mimi.new(cfg, vb)
}

public fun loadMimiB(
    batchSize: Int?,
    modelFile: String,
    numCodebooks: Int?,
    dev: Device,
): Mimi {
    val vb = VarBuilder.fromMmapedSafetensors(listOf(modelFile), DType.F32, dev)
    val cfg = MimiConfig.v01(numCodebooks)
    return Mimi.newInternal(batchSize, cfg, vb)
}
