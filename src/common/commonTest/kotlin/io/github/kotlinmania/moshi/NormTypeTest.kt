package io.github.kotlinmania.moshi

import io.github.kotlinmania.serdejson.fromStr
import io.github.kotlinmania.serdejson.toStr
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NormTypeTest {
    @Test
    fun testNormTypeValues() {
        val values = NormType.entries
        assertEquals(2, values.size)
        assertTrue(values.contains(NormType.RmsNorm))
        assertTrue(values.contains(NormType.LayerNorm))
    }

    @Test
    fun testNormTypeSerdeRoundtrip() {
        for (norm in NormType.entries) {
            val json = toStr(norm).getOrThrow()
            assertEquals("\"${norm.name}\"", json)
            val decoded = fromStr(json, NormType).getOrThrow()
            assertEquals(norm, decoded)
        }
    }
}
