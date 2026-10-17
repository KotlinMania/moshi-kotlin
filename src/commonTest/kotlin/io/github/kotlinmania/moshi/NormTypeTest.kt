package io.github.kotlinmania.moshi

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
}
