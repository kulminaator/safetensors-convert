package stconv

import (
	"strings"
	"testing"
)

// Real tensor names from the bundled Qwen3.5 models (see README, "Regression
// testing with the bundled model inputs"), plus names from other common
// families, that the default policy must keep at their original dtype for
// every target.
var hardProtectedNames = []string{
	"model.language_model.layers.0.input_layernorm.weight",
	"model.language_model.layers.0.post_attention_layernorm.weight",
	"model.language_model.norm.weight",
	"model.language_model.layers.0.self_attn.q_norm.weight",
	"model.language_model.layers.0.self_attn.k_norm.weight",
	"model.language_model.layers.0.linear_attn.norm.weight",
	"model.visual.blocks.0.norm1.weight",
	"model.visual.blocks.0.norm2.weight",
	"mtp.pre_fc_norm_embedding.weight",
	"mtp.pre_fc_norm_hidden.weight",
	"mtp.layers.0.input_layernorm.weight",
	"model.language_model.embed_tokens.weight",
	"model.embed_tokens.weight",
	"transformer.hf.embeddings.word_embeddings.weight",
	"llama.model.tok_embeddings.weight",
	"gpt2.wte",
	"gpt2.model.embeddings.weight",
	"lm_head",
	"lm_head.weight",
	"model.lm_head.weight",
	"gpt2.lm_head",
	"output_layer.weight",
	// Bias tensors: hard-protected for every target (small 1-D additive
	// vectors; quantizing them saves nothing and a per-tensor scale on
	// them degrades the whole vector).
	"model.language_model.layers.0.input_layernorm.bias",
	"model.language_model.layers.0.self_attn.q_proj.bias",
	"model.visual.blocks.0.attn.proj.bias",
	"model.visual.blocks.0.attn.qkv.bias",
	"model.visual.blocks.0.mlp.linear_fc1.bias",
	"model.visual.blocks.0.mlp.linear_fc2.bias",
	"model.visual.blocks.0.norm1.bias",
	"model.visual.blocks.0.norm2.bias",
	"model.visual.merger.norm.bias",
	"model.visual.patch_embed.proj.bias",
	// Underscore-named bias parameters (linear-attention dt_bias).
	"model.language_model.layers.0.linear_attn.dt_bias",
	"bias",
	// Small linear-attention (delta-net) control parameters.
	"model.language_model.layers.0.linear_attn.A_log",
	"model.language_model.layers.0.linear_attn.in_proj_a.weight",
	"model.language_model.layers.0.linear_attn.in_proj_b.weight",
	"model.language_model.layers.0.linear_attn.conv1d.weight",
}

// Attention projections: protected only for the unscaled fp8 targets.
// Exactly the q/k/v/o_proj set from quantization-advice.md section 6 -
// the fused QKV and out_proj forms are NOT in it (see unprotectedNames).
var fp8OnlyProtectedNames = []string{
	"model.language_model.layers.0.self_attn.q_proj.weight",
	"model.language_model.layers.0.self_attn.k_proj.weight",
	"model.language_model.layers.0.self_attn.v_proj.weight",
	"model.language_model.layers.0.self_attn.o_proj.weight",
}

// Names the policy must never touch, for any target.
var unprotectedNames = []string{
	"model.language_model.layers.0.mlp.down_proj.weight",
	"model.language_model.layers.0.mlp.gate_proj.weight",
	"model.language_model.layers.0.mlp.up_proj.weight",
	"model.visual.patch_embed.proj.weight",
	"model.visual.pos_embed.weight",
	// Fused QKV and the linear-attention output projection are not in
	// the attention-projection protection set: they convert for every
	// target, including the unscaled fp8 ones.
	"model.visual.blocks.0.attn.qkv.weight",
	"model.language_model.layers.1.linear_attn.in_proj_qkv.weight",
	"model.language_model.layers.1.linear_attn.out_proj.weight",
}

func TestProtectDefaultHardProtected(t *testing.T) {
	for _, name := range hardProtectedNames {
		for _, target := range allTargets {
			got, reason := ProtectDefault(name, target)
			if !got {
				t.Errorf("ProtectDefault(%q, %v) = false, want protected", name, target)
				continue
			}
			if reason == "" {
				t.Errorf("ProtectDefault(%q, %v) protected with empty reason", name, target)
			}
		}
	}
}

func TestProtectDefaultFP8Only(t *testing.T) {
	for _, name := range fp8OnlyProtectedNames {
		for _, target := range []TargetKind{TargetFP8E4M3, TargetFP8E5M2} {
			if got, reason := ProtectDefault(name, target); !got || reason == "" {
				t.Errorf("ProtectDefault(%q, %v) = %v, %q; want protected with reason", name, target, got, reason)
			}
		}
		for _, target := range []TargetKind{TargetInt8, TargetInt8ConvRot, TargetMxFP4, TargetNVFP4, TargetInt4} {
			if got, reason := ProtectDefault(name, target); got {
				t.Errorf("ProtectDefault(%q, %v) = true, %q; want converted (scaled target)", name, target, reason)
			}
		}
	}
}

func TestProtectDefaultUnprotected(t *testing.T) {
	for _, name := range unprotectedNames {
		for _, target := range allTargets {
			if got, reason := ProtectDefault(name, target); got {
				t.Errorf("ProtectDefault(%q, %v) = true, %q; want not protected", name, target, reason)
			}
		}
	}
}

func TestProtectDefaultReasons(t *testing.T) {
	cases := []struct {
		name   string
		target TargetKind
		reason string
	}{
		{"model.language_model.norm.weight", TargetFP8E4M3, protectReasonNorms},
		{"model.visual.blocks.0.norm1.bias", TargetFP8E4M3, protectReasonBias},
		{"model.language_model.layers.0.linear_attn.A_log", TargetFP8E4M3, protectReasonLinAttn},
		{"model.language_model.embed_tokens.weight", TargetInt8, protectReasonEmbeddings},
		{"lm_head.weight", TargetMxFP4, protectReasonLMHead},
		{"model.language_model.layers.0.self_attn.q_proj.weight", TargetFP8E4M3, protectReasonAttnProj},
	}
	for _, c := range cases {
		got, reason := ProtectDefault(c.name, c.target)
		if !got {
			t.Fatalf("ProtectDefault(%q, %v) = false, want protected", c.name, c.target)
		}
		if reason != c.reason {
			t.Errorf("ProtectDefault(%q, %v) reason = %q, want %q", c.name, c.target, reason, c.reason)
		}
		// Reasons are printed verbatim in the per-tensor report: one line,
		// no trailing punctuation.
		if strings.ContainsAny(reason, "\n\r") || len(reason) == 0 {
			t.Errorf("reason %q is not a single line", reason)
		}
		if last := reason[len(reason)-1]; last == '.' || last == ',' || last == ';' {
			t.Errorf("reason %q has trailing punctuation", reason)
		}
	}
}
