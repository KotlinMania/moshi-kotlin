# Immediate Actions - High-Value Files

Based on AST analysis, here are the concrete next steps.

## Summary

- **Files Present:** 17/18 (94.4%)
- **Function parity:** 336/368 matched (target 383) — 91.3%
- **Class/type parity:** 62/100 matched (target 125) — 62.0%
- **Combined symbol parity:** 398/468 matched (target 508) — 85.0%
- **Average inline-code cosine:** 0.26 (function body across 17 matched files)
- **Average documentation cosine:** 0.17 (doc text across 17 matched files)
- **Cheat-zeroed Files:** 0
- **Critical Issues:** 17 files with <0.60 function similarity
- **Needs Review:** 0 files with 0.60-0.84 function similarity
- **Excellent:** 0 files with >=0.85 function similarity

## Priority 1: Fix Incomplete High-Dependency Files

No incomplete high-dependency files detected.

## Priority 2: Port Missing High-Value Files

Critical missing files (>10 dependencies):

No missing high-value files detected.

## Detailed Work Items

Every matched file is listed below with function and type symbol parity.

### 1. mimi

- **Target:** `moshi.Mimi`
- **Similarity:** 0.28
- **Dependents:** 1
- **Priority Score:** 1041707.2
- **Functions:** 11/14 matched
- **Missing functions:** `new_`, `load`, `load_b`
- **Types:** 2/3 matched
- **Missing types:** `Config`

### 2. kv_cache

- **Target:** `moshi.KvCache`
- **Similarity:** 0.23
- **Dependents:** 1
- **Priority Score:** 1002407.7
- **Functions:** 20/20 matched (target 23)
- **Missing functions:** _none_
- **Types:** 4/4 matched (target 6)
- **Missing types:** _none_
- **Tests:** 1/1 matched

### 3. transformer

- **Target:** `moshi.Transformer`
- **Similarity:** 0.30
- **Dependents:** 0
- **Priority Score:** 186807.0
- **Functions:** 46/50 matched (target 46)
- **Missing functions:** `flash_attn`, `flash_attn`, `new`, `forward`
- **Types:** 4/18 matched (target 17)
- **Missing types:** `Config`, `XaGate`, `StreamingMultiheadCrossAttention`, `Rope`, `RotaryEmbedding`, `StreamingMultiheadAttention`, `Mlp`, `RmsNorm`, `LayerNorm`, `Norm`, `StreamingTransformerLayer`, `StreamingTransformer`, `ProjectedTransformer`, `Transformer`
- **Lint issues:** 1

### 4. batched_transformer

- **Target:** `moshi.BatchedTransformer`
- **Similarity:** 0.23
- **Dependents:** 0
- **Priority Score:** 173907.7
- **Functions:** 22/31 matched (target 22)
- **Missing functions:** `new`, `new_shortcut`, `forward`, `new`, `forward`, `new`, `forward`, `new`, `forward`
- **Types:** 0/8 matched (target 4)
- **Missing types:** `StreamingMultiheadAttention`, `Mlp`, `RmsNorm`, `LayerNorm`, `Norm`, `StreamingTransformerLayer`, `StreamingTransformer`, `ProjectedTransformer`

### 5. lm

- **Target:** `moshi.Lm`
- **Similarity:** 0.25
- **Dependents:** 0
- **Priority Score:** 96307.5
- **Functions:** 47/54 matched
- **Missing functions:** `load`, `load_streaming`, `load_streaming_both_ways`, `load_vision`, `load_s2s`, `load_asr`, `new`
- **Types:** 7/9 matched (target 12)
- **Missing types:** `Config`, `StreamingTransformer`

### 6. streaming

- **Target:** `moshi.Streaming`
- **Similarity:** 0.14
- **Dependents:** 0
- **Priority Score:** 73908.6
- **Functions:** 25/31 matched (target 33)
- **Missing functions:** `fmt`, `from`, `fmt`, `from`, `from`, `from`
- **Types:** 7/8 matched
- **Missing types:** `Dim`
- **Lint issues:** 1

### 7. lm_generate_multistream

- **Target:** `moshi.LmGenerateMultistream`
- **Similarity:** 0.26
- **Dependents:** 0
- **Priority Score:** 32007.4
- **Functions:** 17/18 matched
- **Missing functions:** `step`
- **Types:** 0/2 matched (target 3)
- **Missing types:** `Config`, `State`

### 8. conv

- **Target:** `moshi.Conv`
- **Similarity:** 0.40
- **Dependents:** 0
- **Priority Score:** 24006.0
- **Functions:** 30/32 matched
- **Missing functions:** `run_conv1d`, `run_conv_tr1d`
- **Types:** 8/8 matched (target 9)
- **Missing types:** _none_
- **Tests:** 2/4 matched
- **Lint issues:** 2

### 9. tts_streaming

- **Target:** `moshi.TtsStreaming`
- **Similarity:** 0.25
- **Dependents:** 0
- **Priority Score:** 22407.5
- **Functions:** 19/19 matched
- **Missing functions:** _none_
- **Types:** 3/5 matched (target 10)
- **Missing types:** `Config`, `State`

### 10. lm_generate

- **Target:** `moshi.LmGenerate`
- **Similarity:** 0.16
- **Dependents:** 0
- **Priority Score:** 21408.4
- **Functions:** 11/11 matched
- **Missing functions:** _none_
- **Types:** 1/3 matched (target 6)
- **Missing types:** `Config`, `State`

### 11. tts

- **Target:** `moshi.Tts`
- **Similarity:** 0.36
- **Dependents:** 0
- **Priority Score:** 20906.4
- **Functions:** 7/7 matched
- **Missing functions:** _none_
- **Types:** 0/2 matched (target 3)
- **Missing types:** `Config`, `Model`

### 12. seanet

- **Target:** `moshi.Seanet`
- **Similarity:** 0.30
- **Dependents:** 0
- **Priority Score:** 12107.0
- **Functions:** 15/15 matched
- **Missing functions:** _none_
- **Types:** 5/6 matched (target 7)
- **Missing types:** `Config`

### 13. conditioner

- **Target:** `moshi.Conditioner`
- **Similarity:** 0.32
- **Dependents:** 0
- **Priority Score:** 11806.8
- **Functions:** 9/9 matched
- **Missing functions:** _none_
- **Types:** 8/9 matched (target 11)
- **Missing types:** `Config`

### 14. asr

- **Target:** `moshi.Asr`
- **Similarity:** 0.27
- **Dependents:** 0
- **Priority Score:** 11707.3
- **Functions:** 14/14 matched (target 15)
- **Missing functions:** _none_
- **Types:** 2/3 matched (target 7)
- **Missing types:** `State`

### 15. quantization

- **Target:** `moshi.Quantization`
- **Similarity:** 0.26
- **Dependents:** 0
- **Priority Score:** 2507.4
- **Functions:** 19/19 matched
- **Missing functions:** _none_
- **Types:** 6/6 matched
- **Missing types:** _none_

### 16. nn

- **Target:** `moshi.Nn`
- **Similarity:** 0.30
- **Dependents:** 0
- **Priority Score:** 2207.0
- **Functions:** 18/18 matched (target 22)
- **Missing functions:** _none_
- **Types:** 4/4 matched (target 6)
- **Missing types:** _none_

### 17. wav

- **Target:** `moshi.Wav`
- **Similarity:** 0.18
- **Dependents:** 0
- **Priority Score:** 708.2
- **Functions:** 6/6 matched (target 24)
- **Missing functions:** _none_
- **Types:** 1/1 matched (target 7)
- **Missing types:** _none_

## Success Criteria

For each file to be considered "complete":
- **Similarity ≥ 0.85** (Excellent threshold)
- All public APIs ported
- All tests ported
- Documentation ported
- port-lint header present

