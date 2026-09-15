# Quantization Advice: Layers That Should Stay BF16

When quantizing a model, do not blindly convert every tensor to the target
data type. Some layers are extremely sensitive to low precision, and
quantizing them causes disproportionate accuracy loss. The default behavior
of the conversion tool should keep the following tensors in BF16 unless the
user explicitly overrides this.

## Layers to keep in BF16

### 1. All LayerNorm weights

LayerNorm parameters (weight and bias) are small per-channel scale factors
applied to activations. They are typically close to 1.0 but not exactly 1.0,
and they encode per-channel corrections learned during training.

Why they are sensitive:

- These values are not large in magnitude, so quantization grids (especially
  FP8 E4M3 with its limited 3-bit mantissa) represent them with coarse
  relative error.
- A small relative error in a scale factor is applied to every activation in
  the channel, so the error propagates to all downstream layers instead of
  being averaged out like it would be for a dense weight matrix.
- The parameter count is tiny (a few floats per hidden dimension), so the
  memory savings from quantizing them are negligible compared to the risk.

### 2. All RMSNorm weights

RMSNorm is the same story as LayerNorm. Modern LLMs use RMSNorm in every
block, and its gain vector multiplies activations by a per-channel scale.

Why it is sensitive:

- RMSNorm has no bias, so the weight is the only learned parameter in the
  normalization. Any error in it directly shifts the activation scale.
- Errors here compound across the stack because every transformer block
  applies its own RMSNorm. Tiny per-block errors add up over dozens or
  hundreds of blocks.
- As with LayerNorm, the storage cost of keeping these in BF16 is
  negligible.

### 3. Normalization weights inside attention (q_norm, k_norm, etc.)

Many recent architectures add fine-grained or query/key specific
normalization inside the attention module (for example, q_norm and k_norm
applied after Q and K projection, or per-head / per-token normalizations).

Why they are sensitive:

- These weights sit directly in the attention score path. The attention
  output is a softmax weighted average, and softmax is extremely sensitive
  to relative shifts in the logits. A small scale error in q or k changes
  the relative ordering of attention weights.
- Attention is a non-linear operation (softmax), so quantization error in
  the normalized Q/K vectors does not average out. It systematically
  distorts where the model attends.
- These norms usually operate on very small feature groups (per head, or
  per token with a small feature size), so there is little redundancy to
  absorb the error.

### 4. Embedding weights (input and output)

The token embedding table maps discrete tokens to vectors, and it is often
shared or mirrored with the final projection.

Why it is sensitive:

- Embeddings are looked up, not averaged. Each token uses exactly one row of
  the table, so there is no averaging effect to mask quantization error.
  The error for a given token is a fixed vector offset, not a noise term.
- Every downstream computation in the network starts from the embedding, so
  its error is present in every layer from the first block onward.
- The embedding table is sparse in usage per input: tokens that appear
  rarely in training data have rows that were barely updated, making them
  fragile to quantization.
- Vocabulary tables are often the single largest parameter block in a model,
  but keeping them in BF16 is usually still far cheaper than the accuracy
  cost of low-precision embeddings, especially for rare tokens.

### 5. Final LM head / output projection

The LM head projects the final hidden state to vocabulary logits. These
logits are fed into a softmax to produce the output distribution.

Why it is sensitive:

- The final logits determine the model's answer. Errors here are not
  corrected by anything, because nothing follows the output layer.
- Logit distributions are often long-tailed. The gap between the top
  candidate and the rest is small, so even a small quantization error can
  flip the argmax or noticeably change the probabilities.
- Quantizing the head changes the temperature of the output distribution in
  a way that is hard to compensate for with calibration.
- If the LM head shares weights with the input embedding, quantizing the
  head means also quantizing the embedding, which compounds the problem.

### 6. Attention projections (q_proj, k_proj, v_proj, o_proj)

These projections are less universally critical than the layers above, but
they are still more sensitive than typical feedforward weights. The decision
depends on the quantization scheme:

- Keep them in BF16 when quantizing to aggressive formats with global or
  large-group scales, or when the model is small, because the error budget
  is tight.
- They may be quantized to FP8 E4M3 only with per-channel (or at least
  per-group with small group sizes) scaling.
- Never quantize them to FP8 E4M3 with global or per-tensor scaling.

Why they are sensitive:

- Q, K, V feed the softmax attention mechanism, which is sensitive to
  relative logit shifts as described above.
- o_proj aggregates the attention output back into the residual stream.
  Errors here affect every token representation that passes through the
  block.
- These matrices often have channels with widely varying activation
  magnitudes (outliers). Without per-channel scales, a single outlier
  channel forces a large global scale, and the rest of the matrix loses
  effective precision. Per-channel scaling fixes this by letting each
  channel use its own range.

The protected set is exactly these four names: q_proj, k_proj, v_proj,
and o_proj. Fused QKV tensors (qkv, in_proj_qkv) and the linear-attention
output projection (out_proj) are deliberately not in it: they are
quantized like any other weight by default.

### 7. Bias tensors

Bias vectors - the additive 1-D bias parameters of attention, MLP,
normalization, and projection layers, named `.bias` or with an
underscore suffix (such as the linear-attention `dt_bias`) - should stay
in BF16:

- They are tiny: one value per output channel, so the memory savings from
  quantizing them are negligible.
- They have no averaging effect. A bias value is added to a single output
  position, so a quantization error in it is a fixed offset on that
  position, not a noise term that a dense weight matrix would average out.
- Under a per-tensor scale (the kind a simple quantizer applies), one
  outlier value in the bias vector sets the scale for the whole vector and
  degrades every other entry; per-channel scaling would buy nothing, since
  there is exactly one value per channel.
- The LayerNorm/RMSNorm biases are already covered by sections 1-2; this
  section extends the same reasoning to all other bias parameters
  (attention, MLP, patch-embed, merger, and friends).

### 8. Small linear-attention parameters (A_log, in_proj_a, in_proj_b, conv1d)

Hybrid LLMs with linear-attention (delta-net) blocks carry a few small
auxiliary parameters that control the recurrent state update instead of
mixing dense features. They should stay at their original dtype (BF16,
or F32 for A_log):

- **A_log** - the per-head decay rate, stored in log space. One value per
  head; a quantization error in log space becomes a *multiplicative* error
  in the decay factor, and it compounds over the whole sequence length
  instead of averaging out.
- **in_proj_a / in_proj_b** - the projections that produce the per-head
  alpha/beta gate values of the state update. Their outputs are tiny
  per-head scalars that decide how much of the incoming information
  replaces the stored state; errors here directly distort the recurrence.
- **conv1d** - the short causal depthwise conv kernel (kernel size of a
  few elements). At most a few tens of thousands of values, applied to
  every position; under a per-tensor scale a single outlier dominates the
  whole kernel.

All of them are small (a few hundred to a few tens of thousands of
elements), so the memory savings from quantizing them are negligible, and
each sits in a high-leverage control path instead of a dense averaging
matrix. The dense projections of the same block (in_proj_qkv, in_proj_z,
out_proj) are large mixing matrices and are deliberately NOT in this set.

## General principles

- Prefer keeping small, high-leverage parameter groups in BF16. The memory
  cost is tiny and the accuracy protection is large.
- Layers with non-linear operations after them (softmax in particular)
  tolerate less quantization error than linear layers.
- Parameters with few elements (norm weights, biases) have no averaging
  effect, so per-element quantization error has no redundancy to hide in.
- When a layer is shared between input and output paths (tie_word_embedding),
  keep it in BF16 because an error in it costs twice.
- If a user explicitly requests a lower precision for one of these layers,
  honor the request, but warn that accuracy may degrade.
