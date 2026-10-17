// port-lint: source conditioner.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cat
import io.github.kotlinmania.serde.SerdeError
import io.github.kotlinmania.serde.SerdeException
import io.github.kotlinmania.serde.SerdeResult
import io.github.kotlinmania.serde.serdeCatching
import io.github.kotlinmania.serdecore.de.Deserialize
import io.github.kotlinmania.serdecore.de.Deserializer
import io.github.kotlinmania.serdecore.de.F32Deserialize
import io.github.kotlinmania.serdecore.de.I32Deserialize
import io.github.kotlinmania.serdecore.de.MapAccess
import io.github.kotlinmania.serdecore.de.StringDeserialize
import io.github.kotlinmania.serdecore.de.Visitor
import kotlin.math.pow

private val LUT_CONFIG_FIELDS: List<String> = listOf("n_bins", "dim", "possible_values")
private val CONTINUOUS_ATTR_FIELDS: List<String> = listOf("dim", "scale_factor", "max_period")
private val CONDITIONER_VARIANTS: List<String> = listOf("Lut", "ContinuousAttribute")

public data class LutConfig(
    public val nBins: Int,
    public val dim: Int,
    public val possibleValues: List<String>,
) {
    public companion object : Deserialize<LutConfig> {
        override fun <D> deserialize(deserializer: D): SerdeResult<LutConfig>
            where D : Deserializer =
            deserializer.deserializeStruct(
                "LutConfig",
                LUT_CONFIG_FIELDS,
                object : Visitor<LutConfig> {
                    override fun expecting(): String = "struct LutConfig"

                    override fun <A> visitMap(access: A): SerdeResult<LutConfig>
                        where A : MapAccess =
                        serdeCatching {
                            var nBins: Int? = null
                            var dim: Int? = null
                            var possibleValues: List<String>? = null
                            val strListDe = StringDeserialize.asListDeserialize()
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "n_bins" -> nBins = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "dim" -> dim = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "possible_values" -> possibleValues = access.nextValueSeed(SeedOf(strListDe)).getOrThrow()
                                    else -> throw SerdeException(SerdeError.unknownField(key, LUT_CONFIG_FIELDS))
                                }
                            }
                            LutConfig(
                                nBins = nBins ?: throw SerdeException(SerdeError.missingField("n_bins")),
                                dim = dim ?: throw SerdeException(SerdeError.missingField("dim")),
                                possibleValues = possibleValues ?: throw SerdeException(SerdeError.missingField("possible_values")),
                            )
                        }
                },
            )
    }
}

public data class ContinuousAttributeConfig(
    public val dim: Int,
    public val scaleFactor: Float,
    public val maxPeriod: Float,
) {
    public companion object : Deserialize<ContinuousAttributeConfig> {
        override fun <D> deserialize(deserializer: D): SerdeResult<ContinuousAttributeConfig>
            where D : Deserializer =
            deserializer.deserializeStruct(
                "ContinuousAttributeConfig",
                CONTINUOUS_ATTR_FIELDS,
                object : Visitor<ContinuousAttributeConfig> {
                    override fun expecting(): String = "struct ContinuousAttributeConfig"

                    override fun <A> visitMap(access: A): SerdeResult<ContinuousAttributeConfig>
                        where A : MapAccess =
                        serdeCatching {
                            var dim: Int? = null
                            var scaleFactor: Float? = null
                            var maxPeriod: Float? = null
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "dim" -> dim = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "scale_factor" -> scaleFactor = access.nextValueSeed(SeedOf(F32Deserialize)).getOrThrow()
                                    "max_period" -> maxPeriod = access.nextValueSeed(SeedOf(F32Deserialize)).getOrThrow()
                                    else -> throw SerdeException(SerdeError.unknownField(key, CONTINUOUS_ATTR_FIELDS))
                                }
                            }
                            ContinuousAttributeConfig(
                                dim = dim ?: throw SerdeException(SerdeError.missingField("dim")),
                                scaleFactor = scaleFactor ?: throw SerdeException(SerdeError.missingField("scale_factor")),
                                maxPeriod = maxPeriod ?: throw SerdeException(SerdeError.missingField("max_period")),
                            )
                        }
                },
            )
    }
}

public sealed class ConditionerConfig {
    public data class Lut(
        public val config: LutConfig,
    ) : ConditionerConfig()

    public data class ContinuousAttribute(
        public val config: ContinuousAttributeConfig,
    ) : ConditionerConfig()

    public companion object : Deserialize<ConditionerConfig> {
        override fun <D> deserialize(deserializer: D): SerdeResult<ConditionerConfig>
            where D : Deserializer =
            deserializer.deserializeMap(
                object : Visitor<ConditionerConfig> {
                    override fun expecting(): String = "internally tagged enum ConditionerConfig"

                    override fun <A> visitMap(access: A): SerdeResult<ConditionerConfig>
                        where A : MapAccess =
                        serdeCatching {
                            var typeTag: String? = null
                            var nBins: Int? = null
                            var dim: Int? = null
                            var possibleValues: List<String>? = null
                            var scaleFactor: Float? = null
                            var maxPeriod: Float? = null
                            val strListDe = StringDeserialize.asListDeserialize()
                            while (true) {
                                val key = access.nextKeySeed(SeedOf(StringDeserialize)).getOrThrow() ?: break
                                when (key) {
                                    "type" -> typeTag = access.nextValueSeed(SeedOf(StringDeserialize)).getOrThrow()
                                    "n_bins" -> nBins = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "dim" -> dim = access.nextValueSeed(SeedOf(I32Deserialize)).getOrThrow()
                                    "possible_values" -> possibleValues = access.nextValueSeed(SeedOf(strListDe)).getOrThrow()
                                    "scale_factor" -> scaleFactor = access.nextValueSeed(SeedOf(F32Deserialize)).getOrThrow()
                                    "max_period" -> maxPeriod = access.nextValueSeed(SeedOf(F32Deserialize)).getOrThrow()
                                    else -> throw SerdeException(SerdeError.custom("unknown field `$key` in ConditionerConfig"))
                                }
                            }
                            when (val tag = typeTag ?: throw SerdeException(SerdeError.missingField("type"))) {
                                "Lut" -> {
                                    Lut(
                                        LutConfig(
                                            nBins = nBins ?: throw SerdeException(SerdeError.missingField("n_bins")),
                                            dim = dim ?: throw SerdeException(SerdeError.missingField("dim")),
                                            possibleValues = possibleValues ?: throw SerdeException(SerdeError.missingField("possible_values")),
                                        ),
                                    )
                                }

                                "ContinuousAttribute" -> {
                                    ContinuousAttribute(
                                        ContinuousAttributeConfig(
                                            dim = dim ?: throw SerdeException(SerdeError.missingField("dim")),
                                            scaleFactor = scaleFactor ?: throw SerdeException(SerdeError.missingField("scale_factor")),
                                            maxPeriod = maxPeriod ?: throw SerdeException(SerdeError.missingField("max_period")),
                                        ),
                                    )
                                }

                                else -> {
                                    throw SerdeException(SerdeError.unknownVariant(tag, CONDITIONER_VARIANTS))
                                }
                            }
                        }
                },
            )
    }
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
