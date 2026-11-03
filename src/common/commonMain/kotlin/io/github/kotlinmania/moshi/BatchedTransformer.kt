// port-lint: source batched_transformer.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candlenn.softmaxLastDim
import kotlin.math.min
import kotlin.math.pow

public class BatchedStreamingMultiheadAttention(
    private val inProj: MaybeQuantizedLinear,
    private val outProj: MaybeQuantizedLinear,
    private val kvRepeat: Int,
    private val numHeads: Int,
    private val context: Int,
    internal var kvCache: ScatteredKvCache,
) {
    public companion object {
        public fun new(
            cfg: TransformerConfig,
            builder: ScatteredCacheBuilder,
            vb: MaybeQuantizedVarBuilder,
        ): BatchedStreamingMultiheadAttention {
            val embedDim = cfg.dModel
            val headDim = embedDim / cfg.numHeads
            val numKv = cfg.numHeads / cfg.kvRepeat
            val outDim = embedDim + 2 * numKv * (embedDim / cfg.numHeads)
            val inProjWeight = vb.get(Shape(outDim, embedDim), "in_proj_weight")
            val inProjBias = if (cfg.biasAttn) vb.getUnquantized(outDim, "in_proj_bias") else null
            val inProj = linearFrom(inProjWeight, inProjBias)
            val outProj = linear(embedDim, embedDim, cfg.biasAttn, vb.pp("out_proj"))
            return BatchedStreamingMultiheadAttention(
                inProj = inProj,
                outProj = outProj,
                kvRepeat = cfg.kvRepeat,
                numHeads = cfg.numHeads,
                context = cfg.context,
                kvCache = builder.makeCache(numKv, headDim),
            )
        }
    }

    public fun isQuantized(): Boolean =
        when (inProj) {
            is MaybeQuantizedLinear.Quantized -> true
            is MaybeQuantizedLinear.Real -> false
        }

    public fun forward(
        xs: Tensor,
        rope: Rope?,
        iam: IndicesAndMask,
    ): Tensor {
        if (kvRepeat != 1) {
            bail("only kv-repeat = 1 is supported")
        }
        val (b, t, hd) = xs.dims3()
        val headDim = hd / numHeads
        val rawQkv = xs.apply(inProj).reshape(Shape(b, t, 3, numHeads, headDim))
        val originalDtype = rawQkv.dtype()
        val qkv = if (isQuantized()) rawQkv.toDtype(matmulDtype(xs.device())) else rawQkv
        var q =
            qkv
                .narrow(2, 0, 1)
                .squeeze(2)
                .transpose(1, 2)
                .contiguous()
        var k =
            qkv
                .narrow(2, 1, 1)
                .squeeze(2)
                .transpose(1, 2)
                .contiguous()
        val v =
            qkv
                .narrow(2, 2, 1)
                .squeeze(2)
                .transpose(1, 2)
                .contiguous()
        if (rope != null) {
            q = rope.applyRotaryEmb(q)
            k = rope.applyRotaryEmb(k)
        }
        val kvPair = kvCache.append(k.contiguous(), v.contiguous(), iam)
        val cachedK = kvPair.k
        val cachedV = kvPair.v
        val kLen = cachedK.dim(2)
        val kTargetLen = t + min(context, kLen - t)
        val (trimmedK, trimmedV) =
            if (kTargetLen < kLen) {
                cachedK.narrow(2, kLen - kTargetLen, kTargetLen) to
                    cachedV.narrow(2, kLen - kTargetLen, kTargetLen)
            } else {
                cachedK to cachedV
            }
        val preWs =
            q
                .matmul(trimmedK.t())
                .affine(headDim.toDouble().pow(-0.5), 0.0)
                .broadcastAdd(iam.mask())
        val ws = softmaxLastDim(preWs)
        val attnOut = ws.matmul(trimmedV)
        return attnOut
            .transpose(1, 2)
            .reshape(Shape(b, t, hd))
            .toDtype(originalDtype)
            .apply(outProj)
    }

    public fun setKvCache(kvCache: ScatteredKvCache) {
        this.kvCache = kvCache
    }
}

public class BatchedStreamingTransformerLayer(
    internal val selfAttn: BatchedStreamingMultiheadAttention,
    private val mlp: Mlp,
    private val norm1: TransformerNorm,
    private val norm2: TransformerNorm,
    private val layerScale1: LayerScale?,
    private val layerScale2: LayerScale?,
    internal val crossAttn: CrossAttentionBlock?,
    private val normFirst: Boolean,
) {
    public companion object {
        public fun new(
            cfg: TransformerConfig,
            builder: ScatteredCacheBuilder,
            vb: MaybeQuantizedVarBuilder,
            sharedCaVb: MaybeQuantizedVarBuilder? = null,
        ): BatchedStreamingTransformerLayer {
            if (cfg.useConvBlock) {
                bail("conv-block is not supported")
            }
            val dModel = cfg.dModel
            val mlp = Mlp.new(cfg, vb)
            val norm1 = TransformerNorm.new(dModel, cfg, vb.pp("norm1"))
            val norm2 = TransformerNorm.new(dModel, cfg, vb.pp("norm2"))
            val layerScale1 = cfg.layerScale?.let { ls -> LayerScale.new(dModel, ls, vb.pp("layer_scale_1")) }
            val layerScale2 = cfg.layerScale?.let { ls -> LayerScale.new(dModel, ls, vb.pp("layer_scale_2")) }
            val selfAttn = BatchedStreamingMultiheadAttention.new(cfg, builder, vb.pp("self_attn"))
            val crossAttn =
                cfg.crossAttention?.let { caCfg ->
                    val normCross = TransformerNorm.newShortcut(dModel, caCfg.norm, vb.pp("norm_cross"))
                    val ca =
                        if (sharedCaVb == null) {
                            StreamingMultiheadCrossAttention.new(cfg, vb.pp("cross_attention"), null)
                        } else {
                            StreamingMultiheadCrossAttention.new(
                                cfg,
                                sharedCaVb.pp("cross_attention"),
                                vb.pp("cross_attention.gate"),
                            )
                        }
                    CrossAttentionBlock(normCross, ca)
                }
            return BatchedStreamingTransformerLayer(
                selfAttn = selfAttn,
                mlp = mlp,
                norm1 = norm1,
                norm2 = norm2,
                layerScale1 = layerScale1,
                layerScale2 = layerScale2,
                crossAttn = crossAttn,
                normFirst = cfg.normFirst,
            )
        }
    }

    public fun forward(
        xs: Tensor,
        rope: Rope?,
        caSrc: CaSrc?,
        iam: IndicesAndMask,
    ): Tensor {
        if (!normFirst) {
            bail("only norm_first = true is supported")
        }
        val n1 = xs.apply(norm1)
        val attnOut = selfAttn.forward(n1, rope, iam)
        val scaledAttnOut = if (layerScale1 != null) attnOut.apply(layerScale1) else attnOut
        var cur = xs.add(scaledAttnOut)

        if (crossAttn != null && caSrc != null) {
            val residual = cur
            val normCrossOut = cur.apply(crossAttn.norm)
            cur = residual.add(crossAttn.attn.forward(normCrossOut, caSrc, null))
        }

        val mlpOut = cur.apply(norm2).apply(mlp)
        val scaledMlpOut = if (layerScale2 != null) mlpOut.apply(layerScale2) else mlpOut
        return cur.add(scaledMlpOut)
    }

    public fun setKvCache(kvCache: ScatteredKvCache) {
        selfAttn.setKvCache(kvCache)
    }
}

public class BatchedStreamingTransformer(
    private val layers: List<BatchedStreamingTransformerLayer>,
    private val positionalEmbedding: PositionalEmbedding,
    private val causal: Boolean,
    private val builder: ScatteredCacheBuilder,
    private val rope: RotaryEmbedding?,
) : StreamingModule {
    public companion object {
        public fun new(
            batchSize: Int,
            cfg: TransformerConfig,
            vb: MaybeQuantizedVarBuilder,
        ): BatchedStreamingTransformer {
            val vbL = vb.pp("layers")
            val rope =
                when (cfg.positionalEmbedding) {
                    PositionalEmbedding.Rope -> {
                        RotaryEmbedding.new(
                            dim = cfg.dModel / cfg.numHeads,
                            theta = cfg.maxPeriod.toFloat(),
                            dev = vb.device(),
                        )
                    }

                    PositionalEmbedding.None, PositionalEmbedding.Sin -> {
                        null
                    }
                }
            val layers = ArrayList<BatchedStreamingTransformerLayer>(cfg.numLayers)
            val builder = ScatteredCacheBuilder.new(batchSize, cfg.context, vb.dtype(), vb.device())
            for (layerIdx in 0 until cfg.numLayers) {
                val sharedVb = if (cfg.sharedCrossAttn) vbL.pp(0) else null
                val layer = BatchedStreamingTransformerLayer.new(cfg, builder, vbL.pp(layerIdx), sharedVb)
                layers.add(layer)
            }
            return BatchedStreamingTransformer(
                layers = layers,
                positionalEmbedding = cfg.positionalEmbedding,
                causal = cfg.causal,
                builder = builder,
                rope = rope,
            )
        }
    }

    public fun forward(
        xs: Tensor,
        m: StreamMask,
    ): Tensor = forwardCa(xs, null, m)

    public fun batchSize(): Int = builder.batchSize()

    private fun positions(): IntArray = builder.positions()

    public fun forwardCa(
        xs: Tensor,
        caSrc: CaSrc?,
        m: StreamMask,
    ): Tensor {
        val (b, t, _) = xs.dims3()
        if (b != batchSize()) {
            bail("unexpected batch size $b != ${batchSize()}")
        }
        if (!causal) {
            bail("only causal mode is supported")
        }
        val cpuMask = m.cpu() ?: bail("batched-transformer expects a mask")
        val iam = builder.indicesAndMask(t, cpuMask)
        val currentRope =
            rope?.let { r ->
                val curPositions = positions()
                val posFlat = UIntArray(b * t)
                for (bIdx in 0 until b) {
                    val v = curPositions[bIdx]
                    for (tIdx in 0 until t) {
                        posFlat[bIdx * t + tIdx] = (v + tIdx).toUInt()
                    }
                }
                val posTensor = Tensor.fromSlice(posFlat, Shape(b, t), xs.device())
                r.rope(posTensor)
            }
        var cur =
            when (positionalEmbedding) {
                PositionalEmbedding.Rope, PositionalEmbedding.None -> xs
                PositionalEmbedding.Sin -> bail("sin positional embedding is not supported")
            }
        for (layer in layers) {
            cur = layer.forward(cur, currentRope, caSrc, iam)
        }
        return cur
    }

    public fun maybePrecomputeCaKv(caSrc: CaSrc?): CaSrc? =
        when (caSrc) {
            null -> {
                null
            }

            is CaSrc.KeysValues -> {
                caSrc
            }

            is CaSrc.Tokens -> {
                if (layers.isEmpty()) {
                    caSrc
                } else {
                    val caBlock = layers[0].crossAttn
                    if (caBlock == null) {
                        caSrc
                    } else {
                        val kv = caBlock.attn.computeKv(caSrc)
                        CaSrc.KeysValues(kv.k, kv.v)
                    }
                }
            }
        }

    public fun copyState(from: BatchedStreamingTransformer) {
        if (layers.size != from.layers.size) {
            bail("cannot copy kv-caches as the transformers have different depths")
        }
        for (i in layers.indices) {
            layers[i].setKvCache(
                from.layers[i]
                    .selfAttn.kvCache
                    .copy(),
            )
        }
    }

    public fun resetBatchIdx(batchIdx: Int) {
        if (batchIdx >= batchSize()) {
            bail("batch_idx $batchIdx is out of bounds for last_reset_pos")
        }
        builder.resetBatchIndex(batchIdx)
    }

    override fun resetState() {
        builder.reset()
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor {
        val t = xs.asOption() ?: return StreamTensor.empty()
        return StreamTensor.fromTensor(forward(t, mask))
    }
}

public class BatchedProjectedTransformer(
    private val transformer: BatchedStreamingTransformer,
    private val inputProj: MaybeQuantizedLinear?,
    private val outputProjs: List<MaybeQuantizedLinear?>,
    private val convLayout: Boolean,
) : StreamingModule {
    public companion object {
        public fun new(
            inputDim: Int,
            outputDims: IntArray,
            batchSize: Int,
            cfg: TransformerConfig,
            vb: MaybeQuantizedVarBuilder,
        ): BatchedProjectedTransformer {
            val transformer = BatchedStreamingTransformer.new(batchSize, cfg, vb.pp("transformer"))
            val inputProj =
                if (inputDim == cfg.dModel) {
                    null
                } else {
                    linear(inputDim, cfg.dModel, false, vb.pp("input_proj"))
                }
            val outputProjs = ArrayList<MaybeQuantizedLinear?>(outputDims.size)
            val vbO = vb.pp("output_projs")
            for ((i, outputDim) in outputDims.withIndex()) {
                val outputProj =
                    if (outputDim == cfg.dModel) {
                        null
                    } else {
                        linear(cfg.dModel, outputDim, false, vbO.pp(i))
                    }
                outputProjs.add(outputProj)
            }
            return BatchedProjectedTransformer(
                transformer = transformer,
                inputProj = inputProj,
                outputProjs = outputProjs,
                convLayout = cfg.convLayout,
            )
        }
    }

    public fun forward(
        xs: Tensor,
        m: StreamMask,
    ): List<Tensor> {
        var cur = if (convLayout) xs.transpose(1, 2) else xs
        if (inputProj != null) {
            cur = cur.apply(inputProj)
        }
        cur = transformer.forward(cur, m)
        val ys = ArrayList<Tensor>(outputProjs.size)
        for (outputProj in outputProjs) {
            var y = if (outputProj != null) cur.apply(outputProj) else cur
            if (convLayout) {
                y = y.transpose(1, 2)
            }
            ys.add(y)
        }
        return ys
    }

    public fun resetBatchIdx(batchIdx: Int) {
        transformer.resetBatchIdx(batchIdx)
    }

    override fun resetState() {
        transformer.resetState()
    }

    override fun step(
        xs: StreamTensor,
        mask: StreamMask,
    ): StreamTensor {
        val raw = xs.asOption() ?: return StreamTensor.empty()
        val transposedIn = if (convLayout) raw.transpose(1, 2) else raw
        val projectedIn = StreamTensor.fromTensor(transposedIn).apply(inputProj)
        val transformed = transformer.step(projectedIn, mask)
        val projectedOut = transformed.apply(outputProjs[0])
        val outOpt = projectedOut.asOption() ?: return StreamTensor.empty()
        val finalOut = if (convLayout) outOpt.transpose(1, 2) else outOpt
        return StreamTensor.fromTensor(finalOut)
    }
}
