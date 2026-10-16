package io.github.kotlinmania.moshi

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NormTypeTest {
    @Test
    fun testNormTypeValues() {
        val values = NormType.values()
        assertEquals(2, values.size)
        assertTrue(values.contains(NormType.RMS_NORM))
        assertTrue(values.contains(NormType.LAYER_NORM))
    }
}
