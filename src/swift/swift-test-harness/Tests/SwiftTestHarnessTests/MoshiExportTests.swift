import Testing
import Moshi

// Smoke test for the Kotlin -> Swift Export -> SPM -> swift test pipeline.
@Suite("Moshi Export Tests")
struct MoshiExportTests {
    @Test("Swift module loads and imports cleanly")
    func swiftModuleLoads() {
        let mimiCfg = MimiConfig.Companion.shared.v01(numCodebooks: nil)
        #expect(mimiCfg.sampleRate == 24000.0)
        let lmCfg = LmConfig.Companion.shared.v01()
        #expect(lmCfg.audioVocabSize == 2049)
        let ttsStreamingCfg = TtsStreamingConfig.Companion.shared.v202501()
        #expect(ttsStreamingCfg.acousticDelay == 2)
    }
}
