// port-lint: tests kv_cache.rs
package io.github.kotlinmania.moshi

import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import kotlin.test.Test
import kotlin.test.assertEquals

class KvCacheTest {
    @Test
    fun testScatteredKvCache() {
        val device = Device.Cpu
        val cache = ScatteredCacheBuilder.new(2, 5, DType.F32, device)
        val inf = Float.POSITIVE_INFINITY

        val iam1 = cache.indicesAndMask(1, listOf(true, false))
        val mask1 = iam1.mask().getOnDim(1, 0).toVec3<Float>()
        assertEquals(listOf(listOf(0u), listOf(0u)), iam1.indices().toVec2<UInt>())
        assertEquals(
            listOf(
                listOf(listOf(0.0f, -inf, -inf, -inf, -inf)),
                listOf(listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
            ),
            mask1,
        )

        val iam2 = cache.indicesAndMask(1, listOf(true, false))
        val mask2 = iam2.mask().getOnDim(1, 0).toVec3<Float>()
        assertEquals(listOf(listOf(1u), listOf(0u)), iam2.indices().toVec2<UInt>())
        assertEquals(
            listOf(
                listOf(listOf(0.0f, 0.0f, -inf, -inf, -inf)),
                listOf(listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
            ),
            mask2,
        )

        val iam3 = cache.indicesAndMask(3, listOf(false, true))
        val mask3 = iam3.mask().getOnDim(1, 0).toVec3<Float>()
        assertEquals(
            listOf(listOf(2u, 2u, 2u), listOf(0u, 1u, 2u)),
            iam3.indices().toVec2<UInt>(),
        )
        assertEquals(
            listOf(
                listOf(
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                ),
                listOf(
                    listOf(0.0f, -inf, -inf, -inf, -inf),
                    listOf(0.0f, 0.0f, -inf, -inf, -inf),
                    listOf(0.0f, 0.0f, 0.0f, -inf, -inf),
                ),
            ),
            mask3,
        )

        val iam4 = cache.indicesAndMask(3, listOf(true, true))
        val mask4 = iam4.mask().getOnDim(1, 0).toVec3<Float>()
        assertEquals(
            listOf(listOf(2u, 3u, 4u), listOf(3u, 4u, 0u)),
            iam4.indices().toVec2<UInt>(),
        )
        assertEquals(
            listOf(
                listOf(
                    listOf(0.0f, 0.0f, 0.0f, -inf, -inf),
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, -inf),
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                ),
                listOf(
                    listOf(-inf, 0.0f, 0.0f, 0.0f, -inf),
                    listOf(-inf, 0.0f, 0.0f, 0.0f, 0.0f),
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                ),
            ),
            mask4,
        )

        val iam5 = cache.indicesAndMask(1, listOf(true, false))
        val mask5 = iam5.mask().getOnDim(1, 0).toVec3<Float>()
        assertEquals(listOf(listOf(0u), listOf(1u)), iam5.indices().toVec2<UInt>())
        assertEquals(
            listOf(
                listOf(listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
                listOf(listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)),
            ),
            mask5,
        )

        val iam6 = cache.indicesAndMask(2, listOf(true, false))
        val mask6 = iam6.mask().getOnDim(1, 0).toVec3<Float>()
        assertEquals(
            listOf(listOf(1u, 2u), listOf(1u, 1u)),
            iam6.indices().toVec2<UInt>(),
        )
        assertEquals(
            listOf(
                listOf(
                    listOf(0.0f, 0.0f, -inf, 0.0f, 0.0f),
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                ),
                listOf(
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                ),
            ),
            mask6,
        )
    }
}
