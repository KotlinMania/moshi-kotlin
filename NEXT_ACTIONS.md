# Immediate Actions - High-Value Files

Based on AST analysis, here are the concrete next steps.

## Summary

- **Files Present:** 10/18 (55.6%)
- **Function parity:** 13/296 matched (target 16) — 4.4%
- **Class/type parity:** 11/100 matched (target 29) — 11.0%
- **Combined symbol parity:** 24/396 matched (target 45) — 6.1%
- **Average inline-code cosine:** 0.09 (function body across 9 matched files)
- **Average documentation cosine:** 0.00 (doc text across 9 matched files)
- **Cheat-zeroed Files:** 6
- **Critical Issues:** 10 files with <0.60 function similarity

## Priority 1: Fix Incomplete High-Dependency Files

No incomplete high-dependency files detected.

## Priority 2: Port Missing High-Value Files

Critical missing files (>10 dependencies):

No missing high-value files detected.

## Detailed Work Items

Every matched file is listed below with function and type symbol parity.

### 1. mimi

- **Target:** `moshi.Mimi [ZERO] [PROVENANCE-FALLBACK]`
- **Similarity:** 0.00
- **Dependents:** 1
- **Priority Score:** 1151610.0
- **Functions:** 0/13 matched (target 0)
- **Missing functions:** `v0_1`, `new`, `batched`, `config`, `encode_pre_quantize`, `encode`, `encode_step`, `decode`, `decode_step`, `reset_state`, `reset_batch_idx`, `load`, `load_b`
- **Types:** 1/3 matched (target 1)
- **Missing types:** `Config`, `Mimi`
- **Provenance warning:** port-lint provenance header matched only after fallback normalization: `moshi-core/src/mimi.rs` vs expected `mimi.rs`
- **Proposed provenance header:** `// port-lint: source mimi.rs` (current: `// port-lint: source moshi-core/src/mimi.rs`)
- **Lint issues:** 1

### 2. lm

- **Target:** `moshi.Lm [ZERO] [PROVENANCE-FALLBACK]`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 505210.0
- **Functions:** 0/43 matched (target 0)
- **Missing functions:** `depformer_cfg`, `v0_1`, `v0_1_vision`, `v0_1_vision_streaming`, `v0_1_streaming`, `v0_1_asr`, `tts_v0_1`, `s2s_v0_1`, `s2s_v0_1_streaming`, `asr_v0_1_1b`, `asr_300m_202501`, `tts_202501`, `s2s_2b_16rvq_202501`, `new`, `forward`, `sample`, `sample_cfg`, `reset_state`, `step`, `reset_batch_idx`, `maybe_precompute_ca_kv`, `forward_ca`, `batched`, `condition_provider`, `in_audio_codebooks`, `audio_pad_token`, `text_start_token`, `generated_audio_codebooks`, `is_quantized`, `device`, `dtype`, `extra_heads`, `forward_cond`, `depformer_sample`, `depformer_sample_cfg`, `load_lm_model`, `load`, `load_streaming`, `load_streaming_both_ways`, `load_vision`, `load_s2s`, `load_asr`, `forced_tokens`
- **Types:** 2/9 matched (target 8)
- **Missing types:** `Config`, `LowRankEmbeddings`, `DepFormerSlice`, `DepFormer`, `StreamingTransformer`, `LmModel`, `ForcedAudioTokens`
- **Provenance warning:** port-lint provenance header matched only after fallback normalization: `moshi-core/src/lm.rs` vs expected `lm.rs`
- **Proposed provenance header:** `// port-lint: source lm.rs` (current: `// port-lint: source moshi-core/src/lm.rs`)
- **Lint issues:** 1

### 3. transformer

- **Target:** `moshi.Transformer [ZERO] [PROVENANCE-FALLBACK]`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 333510.0
- **Functions:** 0/17 matched (target 0)
- **Missing functions:** `new`, `forward`, `is_quantized`, `compute_kv`, `apply_rotary_emb`, `rope`, `flash_attn`, `reset_kv_cache`, `set_kv_cache`, `new_shortcut`, `current_seq_len`, `forward_ca`, `maybe_precompute_ca_kv`, `copy_state`, `reset_batch_idx`, `reset_state`, `step`
- **Types:** 2/18 matched (target 4)
- **Missing types:** `Config`, `CaSrc`, `LayerScale`, `XaGate`, `StreamingMultiheadCrossAttention`, `Rope`, `RotaryEmbedding`, `StreamingMultiheadAttention`, `Mlp`, `RmsNorm`, `LayerNorm`, `Norm`, `StreamingTransformerLayer`, `StreamingTransformer`, `ProjectedTransformer`, `Transformer`
- **Provenance warning:** port-lint provenance header matched only after fallback normalization: `moshi-core/src/transformer.rs` vs expected `transformer.rs`
- **Proposed provenance header:** `// port-lint: source transformer.rs` (current: `// port-lint: source moshi-core/src/transformer.rs`)
- **Lint issues:** 1

### 4. tts_streaming

- **Target:** `moshi.TtsStreaming [PROVENANCE-FALLBACK]`
- **Similarity:** 0.04
- **Dependents:** 0
- **Priority Score:** 202209.6
- **Functions:** 1/17 matched (target 1)
- **Missing functions:** `new`, `step_idx`, `audio_pad_token`, `config`, `step`, `overwrite_last_text_token`, `audio_tokens`, `text_tokens`, `last_audio_tokens`, `audio_codebooks`, `device`, `dtype`, `tokenize_prompt`, `sample_rate`, `encode`, `empty`
- **Types:** 1/5 matched (target 2)
- **Missing types:** `Config`, `State`, `Speaker`, `SpeakerEncoder`
- **Provenance warning:** port-lint provenance header matched only after fallback normalization: `moshi-core/src/tts_streaming.rs` vs expected `tts_streaming.rs`
- **Proposed provenance header:** `// port-lint: source tts_streaming.rs` (current: `// port-lint: source moshi-core/src/tts_streaming.rs`)
- **Lint issues:** 1

### 5. conv

- **Target:** `moshi.Conv [ZERO] [PROVENANCE-FALLBACK]`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 192110.0
- **Functions:** 0/13 matched (target 0)
- **Missing functions:** `conv1d_weight_norm`, `new`, `forward`, `get_extra_padding_for_conv1d`, `pad1d`, `unpad1d`, `reset_batch_idx`, `reset_state`, `step`, `run_conv1d`, `run_conv_tr1d`, `conv1d`, `conv_tr1d`
- **Types:** 2/8 matched (target 2)
- **Missing types:** `NormConv1d`, `NormConvTranspose1d`, `StreamableConv1d`, `StreamableConvTranspose1d`, `ConvDownsample1d`, `ConvTrUpsample1d`
- **Tests:** 0/4 matched
- **Provenance warning:** port-lint provenance header matched only after fallback normalization: `moshi-core/src/conv.rs` vs expected `conv.rs`
- **Proposed provenance header:** `// port-lint: source conv.rs` (current: `// port-lint: source moshi-core/src/conv.rs`)
- **Lint issues:** 1

### 6. asr

- **Target:** `moshi.Asr [ZERO] [PROVENANCE-FALLBACK]`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 151610.0
- **Functions:** 0/13 matched (target 0)
- **Missing functions:** `reset`, `text_token`, `is_first_step`, `next_token`, `new`, `model_step_idx`, `device`, `batch_size`, `asr_delay_in_tokens`, `step_pcm`, `text_tokens`, `step_tokens`, `reset_batch_idx`
- **Types:** 1/3 matched (target 4)
- **Missing types:** `ItemState`, `State`
- **Provenance warning:** port-lint provenance header matched only after fallback normalization: `moshi-core/src/asr.rs` vs expected `asr.rs`
- **Proposed provenance header:** `// port-lint: source asr.rs` (current: `// port-lint: source moshi-core/src/asr.rs`)
- **Lint issues:** 1

### 7. lm_generate_multistream

- **Target:** `moshi.LmGenerateMultistream [PROVENANCE-FALLBACK]`
- **Similarity:** 0.22
- **Dependents:** 0
- **Priority Score:** 131807.8
- **Functions:** 5/16 matched (target 5)
- **Missing functions:** `new`, `step_idx`, `config`, `user_rating`, `set_user_rating`, `apply_repetition_penalty`, `step_`, `step_without_ca_src`, `audio_tokens`, `text_tokens`, `last_audio_tokens`
- **Types:** 0/2 matched (target 1)
- **Missing types:** `Config`, `State`
- **Provenance warning:** port-lint provenance header matched only after fallback normalization: `moshi-core/src/lm_generate_multistream.rs` vs expected `lm_generate_multistream.rs`
- **Proposed provenance header:** `// port-lint: source lm_generate_multistream.rs` (current: `// port-lint: source moshi-core/src/lm_generate_multistream.rs`)
- **Lint issues:** 1

### 8. lm_generate

- **Target:** `moshi.LmGenerate [PROVENANCE-FALLBACK]`
- **Similarity:** 0.18
- **Dependents:** 0
- **Priority Score:** 91308.2
- **Functions:** 3/10 matched (target 3)
- **Missing functions:** `new`, `step_idx`, `config`, `set_audio_tokens`, `step`, `last_text_token`, `last_audio_tokens`
- **Types:** 1/3 matched (target 5)
- **Missing types:** `Config`, `State`
- **Provenance warning:** port-lint provenance header matched only after fallback normalization: `moshi-core/src/lm_generate.rs` vs expected `lm_generate.rs`
- **Proposed provenance header:** `// port-lint: source lm_generate.rs` (current: `// port-lint: source moshi-core/src/lm_generate.rs`)
- **Lint issues:** 1

### 9. wav

- **Target:** `moshi.Wav [PROVENANCE-FALLBACK]`
- **Similarity:** 0.37
- **Dependents:** 0
- **Priority Score:** 10506.3
- **Functions:** 4/4 matched (target 7)
- **Missing functions:** _none_
- **Types:** 0/1 matched
- **Missing types:** `Sample`
- **Provenance warning:** port-lint provenance header matched only after fallback normalization: `moshi-core/src/wav.rs` vs expected `wav.rs`
- **Proposed provenance header:** `// port-lint: source wav.rs` (current: `// port-lint: source moshi-core/src/wav.rs`)
- **Lint issues:** 1

### 10. lib

- **Target:** `moshi.Lib [STUB] [PROVENANCE-FALLBACK]`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 110.0
- **Functions:** 0/0 matched
- **Missing functions:** _none_
- **Types:** 1/1 matched
- **Missing types:** _none_
- **Provenance warning:** port-lint provenance header matched only after fallback normalization: `moshi-core/src/lib.rs` vs expected `lib.rs`
- **Proposed provenance header:** `// port-lint: source lib.rs` (current: `// port-lint: source moshi-core/src/lib.rs`)
- **Lint issues:** 1

## Success Criteria

For each file to be considered "complete":
- **Similarity ≥ 0.85** (Excellent threshold)
- All public APIs ported
- All tests ported
- Documentation ported
- port-lint header present

