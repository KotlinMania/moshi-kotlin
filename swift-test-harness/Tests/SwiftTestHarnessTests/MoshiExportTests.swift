import Testing
import Moshi

// Smoke test for the Kotlin -> Swift Export -> SPM -> swift test pipeline.
@Suite("Moshi Export Tests")
struct MoshiExportTests {
    @Test("Swift module loads and imports cleanly")
    func swiftModuleLoads() {
        #expect(true)
    }
}
