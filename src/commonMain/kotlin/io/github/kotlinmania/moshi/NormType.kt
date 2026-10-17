package io.github.kotlinmania.moshi

import io.github.kotlinmania.serde.SerdeError
import io.github.kotlinmania.serde.SerdeResult
import io.github.kotlinmania.serde.serdeCatching
import io.github.kotlinmania.serdecore.de.Deserialize
import io.github.kotlinmania.serdecore.de.DeserializeSeed
import io.github.kotlinmania.serdecore.de.Deserializer
import io.github.kotlinmania.serdecore.de.EnumAccess
import io.github.kotlinmania.serdecore.de.StringDeserialize
import io.github.kotlinmania.serdecore.de.Visitor
import io.github.kotlinmania.serdecore.de.mutableListDeserialize
import io.github.kotlinmania.serdecore.de.nullableDeserialize
import io.github.kotlinmania.serdecore.ser.Serialize
import io.github.kotlinmania.serdecore.ser.Serializer
import io.github.kotlinmania.serdecore.ser.serialize

internal class SeedOf<T>(
    private val deserialize: Deserialize<T>,
) : DeserializeSeed<T> {
    override fun <D> deserialize(deserializer: D): SerdeResult<T>
        where D : Deserializer = deserialize.deserialize(deserializer)
}

internal fun Int.asSerialize(): Serialize =
    object : Serialize {
        override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = this@asSerialize.serialize(serializer)
    }

internal fun UInt.asSerialize(): Serialize =
    object : Serialize {
        override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = this@asSerialize.serialize(serializer)
    }

internal fun Float.asSerialize(): Serialize =
    object : Serialize {
        override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = this@asSerialize.serialize(serializer)
    }

internal fun Double.asSerialize(): Serialize =
    object : Serialize {
        override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = this@asSerialize.serialize(serializer)
    }

internal fun Boolean.asSerialize(): Serialize =
    object : Serialize {
        override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = this@asSerialize.serialize(serializer)
    }

internal fun String.asSerialize(): Serialize =
    object : Serialize {
        override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> = this@asSerialize.serialize(serializer)
    }

internal fun <T : Any> T?.asNullableSerialize(wrap: (T) -> Serialize): Serialize =
    object : Serialize {
        override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
            if (this@asNullableSerialize == null) {
                serializer.serializeNone()
            } else {
                serializer.serializeSome(wrap(this@asNullableSerialize))
            }
    }

internal fun <T> List<T>.asListSerialize(wrap: (T) -> Serialize): Serialize =
    object : Serialize {
        override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
            serdeCatching {
                val seq = serializer.serializeSeq(this@asListSerialize.size).getOrThrow()
                for (item in this@asListSerialize) {
                    seq.serializeElement(wrap(item)).getOrThrow()
                }
                seq.end().getOrThrow()
            }
    }

internal fun <T> Deserialize<T>.asListDeserialize(): Deserialize<List<T>> {
    val mutDe = mutableListDeserialize(this)
    return object : Deserialize<List<T>> {
        override fun <D> deserialize(deserializer: D): SerdeResult<List<T>>
            where D : Deserializer = mutDe.deserialize(deserializer).map { it.toList() }
    }
}

internal fun <T : Any> Deserialize<T>.asNullableDeserialize(): Deserialize<T?> = nullableDeserialize(this)

private val NORM_TYPE_VARIANTS: List<String> = listOf("RmsNorm", "LayerNorm")

public enum class NormType : Serialize {
    RmsNorm,
    LayerNorm,
    ;

    override fun <Ok> serialize(serializer: Serializer<Ok>): SerdeResult<Ok> =
        when (this) {
            RmsNorm -> serializer.serializeUnitVariant("NormType", 0u, "RmsNorm")
            LayerNorm -> serializer.serializeUnitVariant("NormType", 1u, "LayerNorm")
        }

    public companion object : Deserialize<NormType> {
        override fun <D> deserialize(deserializer: D): SerdeResult<NormType>
            where D : Deserializer =
            deserializer.deserializeEnum(
                "NormType",
                NORM_TYPE_VARIANTS,
                object : Visitor<NormType> {
                    override fun expecting(): String = "enum NormType"

                    override fun visitStr(v: String): SerdeResult<NormType> =
                        when (v) {
                            "RmsNorm" -> SerdeResult.success(RmsNorm)
                            "LayerNorm" -> SerdeResult.success(LayerNorm)
                            else -> SerdeResult.failure(SerdeError.unknownVariant(v, NORM_TYPE_VARIANTS))
                        }

                    override fun visitString(v: String): SerdeResult<NormType> = visitStr(v)

                    override fun <A> visitEnum(access: A): SerdeResult<NormType>
                        where A : EnumAccess =
                        serdeCatching {
                            val (variant, variantAccess) = access.variantSeed(SeedOf(StringDeserialize)).getOrThrow()
                            variantAccess.unitVariant().getOrThrow()
                            visitStr(variant).getOrThrow()
                        }
                },
            )
    }
}
