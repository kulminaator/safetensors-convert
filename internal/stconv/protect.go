// protect.go holds the built-in default precision policy: a set of tensor
// name patterns that should stay at their original dtype by default, even
// when a conversion target is set. The reasoning for each category is in
// quantization-advice.md at the repo root; the patterns here are the
// mechanical encoding of that advice.
//
// Two categories:
//
//   - hard protection: always kept at the original dtype, for any target.
//     Norm weights (LayerNorm/RMSNorm, including attention-internal
//     q_norm/k_norm and numbered vision norms), token embedding tables,
//     and the final LM head / output projection.
//   - fp8-only protection: kept at the original dtype only when the
//     resolved target is fp8 (e4m3 or e5m2). Attention projections
//     (q/k/v/o_proj and fused qkv variants) feed the softmax attention
//     path and have outlier channels; per quantization-advice.md they
//     must never be fp8 without per-channel scaling, and this tool's fp8
//     applies no scaling at all. Scaled targets (int8, int8_convrot,
//     mxfp4, nvfp4, int4) are allowed.
//
// The policy is a default, not a lock: an explicit config rule for a
// specific tensor name still wins (enforced by the caller), and the whole
// policy can be switched off per run.
package stconv

import "regexp"

// Reason strings are printed verbatim in the per-tensor report (one line,
// no trailing punctuation), so keep them short and stable.
const (
	protectReasonNorms      = "default policy: norm weights stay at original precision"
	protectReasonEmbeddings = "default policy: token embeddings stay at original precision"
	protectReasonLMHead     = "default policy: output head stays at original precision"
	protectReasonAttnProj   = "default policy: attention projections need per-channel scaling, fp8 not applied by default"
)

type protectRule struct {
	re     *regexp.Regexp // compiled once, matched against the full tensor name
	reason string
}

// Hard-protected names. The two norm patterns cover segments ending in
// "norm" (input_layernorm, post_attention_layernorm, q_norm, k_norm, the
// final model norm, numbered vision norm1/norm2, plain "norm") and
// segments with "norm" in the middle (MTP pre_fc_norm_embedding /
// pre_fc_norm_hidden). Positional embeddings (pos_embed) are
// deliberately not protected: they are small buffer tables, not the token
// embedding path the advice covers, and -min-elems already handles them.
var hardProtectRules = []protectRule{
	{regexp.MustCompile(`(^|\.)[a-z0-9_]*norm([0-9]+)?\.weight$`), protectReasonNorms},
	{regexp.MustCompile(`(^|\.)[a-z0-9_]*_norm_[a-z0-9_]*\.weight$`), protectReasonNorms},
	{regexp.MustCompile(`embed_tokens\.weight$`), protectReasonEmbeddings},
	{regexp.MustCompile(`word_embeddings\.weight$`), protectReasonEmbeddings},
	{regexp.MustCompile(`tok_embeddings\.weight$`), protectReasonEmbeddings},
	{regexp.MustCompile(`(^|\.)wte$`), protectReasonEmbeddings},
	{regexp.MustCompile(`\.embeddings?\.weight$`), protectReasonEmbeddings},
	{regexp.MustCompile(`(^|\.)lm_head(\.weight)?$`), protectReasonLMHead},
	{regexp.MustCompile(`output_layer\.weight$`), protectReasonLMHead},
}

// fp8-only protected names: attention projections and their fused forms.
var fp8ProtectRules = []protectRule{
	{regexp.MustCompile(`(^|\.)q_proj\.weight$`), protectReasonAttnProj},
	{regexp.MustCompile(`(^|\.)k_proj\.weight$`), protectReasonAttnProj},
	{regexp.MustCompile(`(^|\.)v_proj\.weight$`), protectReasonAttnProj},
	{regexp.MustCompile(`(^|\.)o_proj\.weight$`), protectReasonAttnProj},
	{regexp.MustCompile(`(^|\.)qkv\.weight$`), protectReasonAttnProj},
	{regexp.MustCompile(`in_proj_qkv\.weight$`), protectReasonAttnProj},
	{regexp.MustCompile(`(^|\.)out_proj\.weight$`), protectReasonAttnProj},
}

// allTargets lists every convertible target; tests use it to assert that
// the hard rules fire for all of them and the fp8-only rules do not.
var allTargets = []TargetKind{
	TargetFP8E4M3, TargetFP8E5M2, TargetInt8, TargetInt8ConvRot,
	TargetMxFP4, TargetNVFP4, TargetInt4,
}

// ProtectDefault reports whether the default policy keeps the named tensor
// at its original dtype instead of converting it to target, and returns a
// one-line human-readable reason (empty when not protected). It is a pure
// name-and-target check: no data is read, so it can run during header
// planning.
func ProtectDefault(name string, target TargetKind) (bool, string) {
	if reason := matchAny(hardProtectRules, name); reason != "" {
		return true, reason
	}
	// Unscaled fp8 is the only format the attention-projection rule
	// refuses; every other target in this tool carries a scale.
	if target == TargetFP8E4M3 || target == TargetFP8E5M2 {
		if reason := matchAny(fp8ProtectRules, name); reason != "" {
			return true, reason
		}
	}
	return false, ""
}

func matchAny(rules []protectRule, name string) string {
	for _, r := range rules {
		if r.re.MatchString(name) {
			return r.reason
		}
	}
	return ""
}
