// port-lint: source conditioner.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.math.pow

@Serializable
public data class LutConfig(
    @SerialName("n_bins")
    public val nBins: Int,
    public val dim: Int,
    @SerialName("possible_values")
    public val possibleValues: List<String>,
)

@Serializable
public data class ContinuousAttributeConfig(
    public val dim: Int,
    @SerialName("scale_factor")
    public val scaleFactor: Float,
    @SerialName("max_period")
    public val maxPeriod: Float,
)

@Serializable
public sealed class ConditionerConfig {
    @Serializable
    @SerialName("Lut")
    public data class Lut(
        public val config: LutConfig,
    ) : ConditionerConfig()

    @Serializable
    @SerialName("ContinuousAttribute")
    public data class ContinuousAttribute(
        public val config: ContinuousAttributeConfig,
    ) : ConditionerConfig()
}

public class LutConditioner(
    private val embed: MaybeQuantizedEmbedding,
    private val outputProj: MaybeQuantizedLinear,
    internal val learntPadding: Tensor,
    private val possibleValues: kotlin.collections.Map<String, Int>,
) {
    public companion object {
        public fun new(
            outputDim: Int,
            cfg: LutConfig,
            vb: MaybeQuantizedVarBuilder,
        ): LutConditioner {
            val embed = MaybeQuantizedEmbedding.new(cfg.nBins + 1, cfg.dim, vb.pp("embed"))
            val outputProj = linear(cfg.dim, outputDim, false, vb.pp("output_proj"))
            val learntPadding = vb.getAsTensor(Shape(1, 1, outputDim), "learnt_padding")
            val possibleValues =
                cfg.possibleValues
                    .mapIndexed { i, v -> v to i }
                    .toMap()
            return LutConditioner(embed, outputProj, learntPadding, possibleValues)
        }
    }

    public fun condition(value: String): Condition {
        val idx =
            possibleValues[value]
                ?: bail("unknown value for lut conditioner '$value'")
        val cond =
            Tensor
                .fromSlice(uintArrayOf(idx.toUInt()), Shape(1, 1), embed.embeddings().device())
                .apply(embed)
                .apply(outputProj)
        return Condition.AddToInput(cond)
    }
}

public class ContinuousAttributeConditioner(
    private val scaleFactor: Float,
    private val maxPeriod: Float,
    private val dim: Int,
    private val outputProj: MaybeQuantizedLinear,
    internal val learntPadding: Tensor,
    private val device: Device,
) {
    public companion object {
        public fun new(
            outputDim: Int,
            cfg: ContinuousAttributeConfig,
            vb: MaybeQuantizedVarBuilder,
        ): ContinuousAttributeConditioner {
            val outputProj = linear(cfg.dim, outputDim, false, vb.pp("output_proj"))
            val learntPadding = vb.getAsTensor(Shape(1, 1, outputDim), "learnt_padding")
            return ContinuousAttributeConditioner(
                scaleFactor = cfg.scaleFactor,
                maxPeriod = cfg.maxPeriod,
                dim = cfg.dim,
                outputProj = outputProj,
                learntPadding = learntPadding,
                device = vb.device(),
            )
        }
    }

    public fun createSinEmbeddings(
        positions: Tensor,
        dtype: DType,
    ): Tensor {
        val dev = positions.device()
        val halfDim = dim / 2
        val positionsDtype = positions.toDtype(dtype)
        val adim =
            FloatArray(halfDim) { i ->
                1.0f / maxPeriod.pow(i.toFloat() / (halfDim - 1).toFloat())
            }
        val adimTensor = Tensor.fromSlice(adim, Shape(1, 1, halfDim), dev)
        val freqs = positionsDtype.broadcastMul(adimTensor)
        return Tensor.cat(listOf(freqs.cos(), freqs.sin()), D.Minus1)
    }

    public fun condition(value: Float): Condition {
        val scaled = (value * scaleFactor).toDouble()
        val positions = Tensor.full(scaled, Shape(1, 1, 1), device)
        val cond =
            createSinEmbeddings(positions, DType.F32)
                .toDtype(outputProj.dtype())
                .apply(outputProj)
        return Condition.AddToInput(cond)
    }
}

public sealed class Conditioner {
    public data class Lut(
        public val conditioner: LutConditioner,
    ) : Conditioner()

    public data class ContinuousAttribute(
        public val conditioner: ContinuousAttributeConditioner,
    ) : Conditioner()
}

public sealed class Condition {
    public data class AddToInput(
        public val tensor: Tensor,
    ) : Condition()
}

public class ConditionProvider(
    private val conditioners: kotlin.collections.Map<String, Conditioner>,
) {
    public companion object {
        public fun new(
            outputDim: Int,
            cfg: kotlin.collections.Map<String, ConditionerConfig>,
            vb: MaybeQuantizedVarBuilder,
        ): ConditionProvider {
            val vbCond = vb.pp("conditioners")
            val conditioners = mutableMapOf<String, Conditioner>()
            for ((conditionerName, conditionerCfg) in cfg.entries) {
                val vbEntry = vbCond.pp(conditionerName)
                val conditioner =
                    when (conditionerCfg) {
                        is ConditionerConfig.Lut -> {
                            Conditioner.Lut(LutConditioner.new(outputDim, conditionerCfg.config, vbEntry))
                        }

                        is ConditionerConfig.ContinuousAttribute -> {
                            Conditioner.ContinuousAttribute(
                                ContinuousAttributeConditioner.new(outputDim, conditionerCfg.config, vbEntry),
                            )
                        }
                    }
                conditioners[conditionerName] = conditioner
            }
            return ConditionProvider(conditioners)
        }
    }

    public fun conditionLut(
        name: String,
        value: String,
    ): Condition =
        when (val c = conditioners[name]) {
            null -> bail("unknown conditioner $name")
            is Conditioner.Lut -> c.conditioner.condition(value)
            is Conditioner.ContinuousAttribute -> bail("cannot use conditioner with a str value $name")
        }

    public fun conditionCont(
        name: String,
        value: Float,
    ): Condition =
        when (val c = conditioners[name]) {
            null -> bail("unknown conditioner $name")
            is Conditioner.ContinuousAttribute -> c.conditioner.condition(value)
            is Conditioner.Lut -> bail("cannot use conditioner with a str value $name")
        }

    public fun learntPadding(name: String): Condition {
        val tensor =
            when (val c = conditioners[name]) {
                null -> bail("unknown conditioner $name")
                is Conditioner.ContinuousAttribute -> c.conditioner.learntPadding
                is Conditioner.Lut -> c.conditioner.learntPadding
            }
        return Condition.AddToInput(tensor)
    }
}
