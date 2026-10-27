# GLM-5.3-Flash GGUF tensor map (Maya-S v2 IQ2_XXS)

**Source artifact:** `GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-0000{1,2,3}-of-00003.gguf`
on `strix-halo-1.local:/home/ugo/software/Maya-data/models/Maya-S-v2-IQ2_XXS/`
(3 shards, 89.83 GiB total tensor data, 1412 tensors).

**Architecture key:** `glm5-next` · **Produced for:** uv-genai/gufo#3 (Phase 0 T0.1)

---

## 1. Geometry (from GGUF metadata — authoritative)

| Parameter | GGUF key | Value |
| --- | --- | --- |
| Blocks | `block_count` | **46** (blk.0–45) |
| Hidden | `embedding_length` | 4096 |
| Dense lead | `leading_dense_block_count` | 3 (blk.0–2) |
| Full-attn layers | `head_count_kv` pattern | **12** DSA: **3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 45** |
| KDA layers | (complement) | **34** |
| Attention heads | `attention.head_count` | 64 |
| KDA head dim | `kda.head_dim` | 128 |
| KDA conv kernel | `ssm.conv_kernel` | 4 |
| KDA gate lower bound | `kda.gate_lower_bound` | −5.0 |
| MLA q_lora | `attention.q_lora_rank` | 1536 |
| MLA kv_lora | `attention.kv_lora_rank` | 512 |
| MLA qk_nope | `attention.key_length_mla` | 256 |
| MLA v | `attention.value_length_mla` | 256 |
| DSA indexer heads | `attention.indexer.head_count` | 32 |
| DSA indexer key len | `attention.indexer.key_length` | 128 |
| DSA top-k | `attention.indexer.top_k` | 2048 |
| DSA k-pool | `attention.indexer.kpool` | 4 |
| MoE experts | `expert_count` | 288 routed + 1 shared |
| MoE top-k | `expert_used_count` | 8 |
| MoE inter | `expert_feed_forward_length` | 2048 |
| MoE shared inter | `expert_shared_feed_forward_length` | 2048 |
| MoE gating | `expert_gating_func` | 2 (sigmoid) |
| MoE scale | `expert_weights_scale` | 2.5 |
| MoE renorm | `expert_weights_norm` | true |
| Dense inter | `feed_forward_length` | 12288 |
| mHC streams | `hyper_connection.count` | 4 |
| mHC Sinkhorn iters | `hyper_connection.sinkhorn_iterations` | 20 |
| MTP layers | `nextn_predict_layers` | 1 |
| Vocab | `vocab_size` | 154880 |
| Context | `context_length` | 1,048,576 |
| RoPE | `rope.dimension_count` | **0** (NoPE) |
| SwiGLU clamp | `swiglu_clamp_exp` / `_shexp` | 10.0 (all layers) |

**Correction to PLAN.md:** the model has **46 blocks** (0–45), not 45.
Block 45 is a real transformer layer (MLA+DSA + MoE) that also hosts the
NextN MTP tensors and — unlike every other block — has **no mHC weights**.

---

## 2. Layer cadence

| Blocks | Attention | FFN | mHC | Notes |
| --- | --- | --- | --- | --- |
| 0–2 | KDA | **dense** (inter 12288) | yes | leading dense |
| 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 | **MLA + DSA** | MoE (288+1) | yes | full attention |
| 4–6, 8–10, 12–14, …, 40–42, 44 | KDA | MoE (288+1) | yes | linear attention |
| **45** | **MLA + DSA** | MoE (288+1) | **no** | + NextN MTP sidecar |

---

## 3. Tensor naming and shapes

GGUF shape convention: `[ne0, ne1, ne2]` where `ne0` is the innermost
(fastest-varying) dimension — this is where quantization blocks run.

### 3.1 Embeddings / output

| Tensor | Shape | Type |
| --- | --- | --- |
| `token_embd.weight` | [4096, 154880] | Q6_K |
| `output.weight` | [4096, 154880] | Q6_K |
| `output_norm.weight` | [4096] | F32 |

### 3.2 Dense attention (KDA layers)

| Tensor | Shape | Type |
| --- | --- | --- |
| `blk.N.attn_norm.weight` | [4096] | F32 |
| `blk.N.attn_q.weight` | [4096, 8192] | Q6_K |
| `blk.N.attn_k.weight` | [4096, 8192] | Q6_K |
| `blk.N.attn_v.weight` | [4096, 8192] | Q6_K |
| `blk.N.attn_output.weight` | [8192, 4096] | Q6_K |

(8192 = 64 heads × 128 head_dim × 1 for Q/K/V; KDA is multi-head over 64×128.)

### 3.3 KDA mixer (`ssm_*`)

| Tensor | Shape | Type |
| --- | --- | --- |
| `blk.N.ssm_a` | [64] | F32 |
| `blk.N.ssm_norm.weight` | [128] | F32 |
| `blk.N.ssm_conv1d_q.weight` | [4, 1, 8192] | F32 |
| `blk.N.ssm_conv1d_k.weight` | [4, 1, 8192] | F32 |
| `blk.N.ssm_conv1d_v.weight` | [4, 1, 8192] | F32 |
| `blk.N.ssm_dt.bias` | [8192] | F32 |
| `blk.N.ssm_beta.weight` | [4096, 64] | Q8_0 |
| `blk.N.ssm_f_a.weight` | [4096, 128] | Q8_0 |
| `blk.N.ssm_f_b.weight` | [128, 8192] | Q8_0 |
| `blk.N.ssm_g_a.weight` | [4096, 128] | Q8_0 |
| `blk.N.ssm_g_b.weight` | [128, 8192] | Q8_0 |

### 3.4 MLA attention (DSA layers)

| Tensor | Shape | Type |
| --- | --- | --- |
| `blk.N.attn_norm.weight` | [4096] | F32 |
| `blk.N.attn_q_a.weight` | [4096, 1536] | Q6_K |
| `blk.N.attn_q_a_norm.weight` | [1536] | F32 |
| `blk.N.attn_q_b.weight` | [1536, 16384] | Q6_K |
| `blk.N.attn_kv_a_mqa.weight` | [4096, 512] | Q6_K |
| `blk.N.attn_kv_a_norm.weight` | [512] | F32 |
| `blk.N.attn_k_b.weight` | [256, 512, 64] | Q6_K |
| `blk.N.attn_v_b.weight` | [512, 256, 64] | Q6_K |
| `blk.N.attn_output.weight` | [16384, 4096] | Q6_K |

(16384 = 64 heads × 256; q_lora=1536 → q_b produces 64×256.)

### 3.5 DSA indexer (DSA layers only)

| Tensor | Shape | Type |
| --- | --- | --- |
| `blk.N.indexer.attn_q_b.weight` | [1536, 4096] | Q8_0 |
| `blk.N.indexer.attn_k.weight` | [4096, 128] | Q8_0 |
| `blk.N.indexer.k_norm.weight` | [128] | F32 |
| `blk.N.indexer.k_norm.bias` | [128] | F32 |
| `blk.N.indexer.proj.weight` | [4096, 32] | Q8_0 |
| `blk.N.indexer_compressor_ape.weight` | [128, 4] | F32 |
| `blk.N.indexer_compressor_gate.weight` | [4096, 128] | Q8_0 |

### 3.6 mHC residual mixing (all blocks except blk.45)

| Tensor | Shape | Type |
| --- | --- | --- |
| `blk.N.hc_attn_base.weight` | [24] | F32 |
| `blk.N.hc_attn_fn.weight` | [16384, 24] | F32 |
| `blk.N.hc_attn_scale.weight` | [3] | F32 |
| `blk.N.hc_ffn_base.weight` | [24] | F32 |
| `blk.N.hc_ffn_fn.weight` | [16384, 24] | F32 |
| `blk.N.hc_ffn_scale.weight` | [3] | F32 |

(24 = 4 streams × 4 streams − 4 + 4? or 4×6; 16384 = 4×4096 hidden concat.)

### 3.7 Dense FFN (blk.0–2 only)

| Tensor | Shape | Type |
| --- | --- | --- |
| `blk.N.ffn_norm.weight` | [4096] | F32 |
| `blk.N.ffn_gate.weight` | [4096, 12288] | Q6_K |
| `blk.N.ffn_up.weight` | [4096, 12288] | Q6_K |
| `blk.N.ffn_down.weight` | [12288, 4096] | Q6_K |

### 3.8 MoE FFN (blk.3–45)

| Tensor | Shape | Type (typical) |
| --- | --- | --- |
| `blk.N.ffn_norm.weight` | [4096] | F32 |
| `blk.N.ffn_gate_inp.weight` | [4096, 288] | F32 |
| `blk.N.exp_probs_b.bias` | [288] | F32 |
| `blk.N.ffn_gate_exps.weight` | [4096, 2048, 288] | IQ2_XXS |
| `blk.N.ffn_up_exps.weight` | [4096, 2048, 288] | IQ2_XXS |
| `blk.N.ffn_down_exps.weight` | [2048, 4096, 288] | IQ2_S / IQ3_XXS / Q3_K |
| `blk.N.ffn_gate_shexp.weight` | [4096, 2048] | Q6_K |
| `blk.N.ffn_up_shexp.weight` | [4096, 2048] | Q6_K |
| `blk.N.ffn_down_shexp.weight` | [2048, 4096] | Q6_K |

**Expert quant variation (down projection):**

| Layers | `ffn_down_exps` | `ffn_gate_exps` / `ffn_up_exps` |
| --- | --- | --- |
| 3–6 (first 4 MoE) | IQ3_XXS | IQ2_XXS |
| 7–40 | IQ2_S | IQ2_XXS |
| 41–44 (last 4 MoE excl. 45) | IQ3_XXS | IQ2_XXS |
| **45** (MTP layer) | **Q3_K** | **Q2_K** |

### 3.9 NextN MTP (blk.45 only)

| Tensor | Shape | Type |
| --- | --- | --- |
| `blk.45.nextn.enorm.weight` | [4096] | F32 |
| `blk.45.nextn.hnorm.weight` | [4096] | F32 |
| `blk.45.nextn.eh_proj.weight` | [8192, 4096] | Q6_K |
| `blk.45.nextn.shared_head_norm.weight` | [4096] | F32 |

(`eh_proj` = [fc_embedding \| fc_hidden] packed across rows, 4096+4096=8192.)

---

## 4. High-precision vs quantized split

Per Maya `modules_to_not_convert` convention (confirmed by types in artifact):

| Always high-precision | Quantized |
| --- | --- |
| All `F32` norms, biases, `ssm_a`, `ssm_conv1d_*`, `ssm_dt.bias`, `hc_*`, `exp_probs_b.bias`, `ffn_gate_inp` (router F32), `indexer.k_norm.*`, `indexer_compressor_ape` | Attention Q/K/V/out (Q6_K), FFN gate/up/down (Q6_K dense / IQ2–IQ3 experts / Q6_K shared), embeddings (Q6_K), KDA low-rank (Q8_0), indexer (Q8_0), MTP eh_proj (Q6_K) |

---

## 5. TP2 column-split alignment matrix (world=2)

**Rule:** gufo `TpPartition` column-splits every expert's intermediate dim.
For gate/up (ne1 = inter): split is on ne1, quant blocks run along ne0 →
**always block-safe**. For down (ne0 = inter): split is on ne0 where quant
blocks live → need `half % block_elems == 0`.

| Tensor pattern | Shape (per expert) | Split dim | Half | Block | Verdict |
| --- | --- | --- | --- | --- | --- |
| dense `ffn_gate/up` (0–2) | [4096, 12288] | ne1 | 6144 | 256 (Q6_K) | **GREEN** |
| dense `ffn_down` (0–2) | [12288, 4096] | ne0 | 6144 | 256 (Q6_K) | **GREEN** (6144%256=0) |
| MoE `ffn_gate_exps` | [4096, 2048] × 288 | ne1 | 1024 | 256 (IQ2_XXS) | **GREEN** |
| MoE `ffn_up_exps` | [4096, 2048] × 288 | ne1 | 1024 | 256 (IQ2_XXS) | **GREEN** |
| MoE `ffn_down_exps` (IQ3_XXS) | [2048, 4096] × 288 | ne0 | 1024 | 256 | **GREEN** (1024%256=0) |
| MoE `ffn_down_exps` (IQ2_S) | [2048, 4096] × 288 | ne0 | 1024 | 256 | **GREEN** |
| MoE `ffn_down_exps` (Q3_K, blk.45) | [2048, 4096] × 288 | ne0 | 1024 | 256 | **GREEN** |
| MoE `ffn_gate/up_shexp` (Q6_K) | [4096, 2048] | ne1 | 1024 | 256 | **GREEN** |
| MoE `ffn_down_shexp` (Q6_K) | [2048, 4096] | ne0 | 1024 | 256 | **GREEN** |
| blk.45 `ffn_gate/up_exps` (Q2_K) | [4096, 2048] × 288 | ne1 | 1024 | 256 | **GREEN** |
| blk.45 `ffn_gate/up_shexp` (Q6_K) | [4096, 2048] | ne1 | 1024 | 256 | **GREEN** |

**All down projections across the artifact:** 89 tensors, **0 RED**.

**Verdict: this Maya-S v2 quant pack is fully TP2 column-split compatible
without re-encode.** Every quant type present (IQ2_XXS, IQ2_S, IQ3_XXS,
Q6_K, Q2_K, Q3_K, Q8_0) uses a 256-element block (except Q8_0 at 32, which
is only on KDA low-rank and indexer — replicated, never split), and all
split points land on block boundaries.

### Types requiring attention if a future quant appears

| Type | Block | Safe halves for inter=2048? | Safe halves for inter=12288? |
| --- | --- | --- | --- |
| IQ2_XXS / IQ2_S / IQ3_XXS / Q2_K / Q3_K / Q4_K / Q5_K / Q6_K | 256 | any multiple of 256 | any multiple of 256 |
| Q8_0 / Q4_0 / Q5_0 | 32 | any multiple of 32 | any multiple of 32 |
| Q2_0 (Maya type 42) | 64 | any multiple of 64 | any multiple of 64 |
| F32 / F16 / BF16 | 1 | always | always |

---

## 6. Metadata keys

All model config under **`glm5-next.*`** prefix (not `qwen4exp.*`).
Tokenizer under standard `tokenizer.ggml.*` keys.

Key metadata (complete list in section 1):

```
general.architecture = glm5-next
glm5-next.block_count = 46
glm5-next.embedding_length = 4096
glm5-next.leading_dense_block_count = 3
glm5-next.expert_count = 288
glm5-next.expert_used_count = 8
glm5-next.expert_feed_forward_length = 2048
glm5-next.expert_shared_count = 1
glm5-next.expert_shared_feed_forward_length = 2048
glm5-next.expert_gating_func = 2          # sigmoid
glm5-next.expert_weights_norm = true
glm5-next.expert_weights_scale = 2.5
glm5-next.feed_forward_length = 12288
glm5-next.hyper_connection.count = 4
glm5-next.hyper_connection.sinkhorn_iterations = 20
glm5-next.kda.gate_lower_bound = -5.0
glm5-next.kda.head_dim = 128
glm5-next.ssm.conv_kernel = 4
glm5-next.attention.head_count = 64
glm5-next.attention.q_lora_rank = 1536
glm5-next.attention.kv_lora_rank = 512
glm5-next.attention.key_length_mla = 256
glm5-next.attention.value_length_mla = 256
glm5-next.attention.indexer.head_count = 32
glm5-next.attention.indexer.key_length = 128
glm5-next.attention.indexer.top_k = 2048
glm5-next.attention.indexer.kpool = 4
glm5-next.nextn_predict_layers = 1
glm5-next.rope.dimension_count = 0        # NoPE
glm5-next.vocab_size = 154880
glm5-next.context_length = 1048576
glm5-next.swiglu_clamp_exp = [10.0 × 46]
glm5-next.swiglu_clamp_shexp = [10.0 × 46]
```

---

## 7. Tokenizer assets in GGUF

| Key | Value |
| --- | --- |
| `tokenizer.ggml.model` | gpt2 |
| `tokenizer.ggml.pre` | glm4 |
| `tokenizer.ggml.bos_token_id` | 154822 |
| `tokenizer.ggml.eos_token_id` | 154820 |
| `tokenizer.ggml.eot_token_id` | 154827 |
| `tokenizer.ggml.eom_token_id` | 154829 |
| `tokenizer.ggml.padding_token_id` | 154820 |
| `tokenizer.ggml.unknown_token_id` | 154820 |
| `tokenizer.chat_template` | starts with `[gMASK]<sop>` |

Chat template framing uses **`[gMASK]<sop>`** control tokens (GLM-4 style).

---

## 8. Byte totals by category

| Category | Tensors | GiB |
| --- | --- | --- |
| Routed MoE experts | 129 | 82.56 |
| Dense attention (KDA Q/K/V/out) | 194 | 4.10 |
| Embeddings + output | 3 | 0.97 |
| Shared MoE experts | 129 | 0.83 |
| MLA projections | 84 | 0.46 |
| Dense FFN (blk.0–2) | 55 | 0.35 |
| Router (gate_inp + exp_probs_b) | 86 | 0.19 |
| mHC | 270 | 0.13 |
| KDA mixer | 374 | 0.13 |
| DSA indexer | 84 | 0.09 |
| NextN MTP | 4 | 0.03 |
| **Total** | **1412** | **89.83** |

## 9. TP2 memory estimate

| Item | Per rank (world=2) |
| --- | --- |
| Routed experts (col-split / 2) | 41.28 GiB |
| Shared experts (col-split / 2) | 0.41 GiB |
| Replicated (attn, KDA, mHC, embed, router, MTP) | 6.44 GiB |
| **Weights total** | **48.14 GiB** |
| KV @ 256K (11.3 KB/token × 11 DSA layers — wait, 12 DSA) | ~3.2 GiB |
| Workspace / scratch | ~2–6 GiB |
| **Estimated total** | **~53–57 GiB** (fits 128 GB Strix Halo with room) |

Note: KV is per-DSA-layer latent `c_kv` 512 floats. With **12** DSA layers
(not 11): 12 × 512 × 2 B = **12.3 KB/token** → 256K ≈ 3.1 GiB, 1M ≈ 12.3 GiB.

**Single-host:** 89.83 GiB weights + KV + scratch does **not** fit Maya-S
on 128 GB with useful context. Use Maya-S24/XS or short context, or TP2.

---

## 10. Loader checklist for `glm5_next`

1. Architecture string: `glm5-next`.
2. Read all geometry from `glm5-next.*` metadata keys (section 1).
3. Per-block tensor set depends on cadence:
   - KDA blocks: `ssm_*` + dense-style `attn_{q,k,v,output}` + `hc_*` (+ FFN).
   - DSA blocks: MLA tensors (`attn_q_a/q_b/kv_a_mqa/k_b/v_b`) + `indexer.*` +
     `indexer_compressor_*` + `hc_*` (+ FFN).
   - blk.45: DSA + MoE + `nextn.*`, **no** `hc_*`.
4. FFN set: dense (0–2) vs MoE (3–45); MoE has `exps` 3D + `shexp` + router.
5. `eh_proj` packs `[fc_embedding | fc_hidden]` across rows (8192 = 2×4096).
6. High-precision list: all F32 tensors + Q8_0 low-rank (section 4).
7. Tokenizer: glm4 pre-tokenizer, gpt2 BPE, `[gMASK]<sop>` chat framing.
