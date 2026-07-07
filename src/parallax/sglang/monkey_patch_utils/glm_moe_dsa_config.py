"""GLM-MoE-DSA (GLM-5.2) config repair for sglang.

transformers >= 5.6 GlmMoeDsaConfig declares `attribute_map = {"head_dim":
"qk_rope_head_dim"}`. GLM-5.2 checkpoints ship BOTH keys in config.json
(head_dim=192, qk_rope_head_dim=64), so loading the json routes head_dim
through the alias and clobbers qk_rope_head_dim to 192. Everything downstream
of the loaded config then disagrees with the checkpoint:

  * fused_qkv_a_proj_with_mqa is built q_lora+kv_lora+qk_rope = 2048+512+192
    = 2752 rows while the checkpoint's q_a_proj+kv_a_proj cat is 2624 →
    `AssertionError param.shape=[2752, 6144] ... loaded_weight.shape=[2624,
    6144]` on the first attention load of ANY pipeline shard;
  * the MLA latent cache is sized kv_lora+192 instead of kv_lora+64.

Dropping the alias keeps `head_dim` as a plain attribute (still 192, which is
what MLA readers expect) and preserves the real qk_rope_head_dim=64.

Additionally, sglang 0.5.12 decides per-layer indexer top-k reuse with the
formula `skip_topk = max(layer_id-1, 0) % index_topk_freq != 0` unless the
config provides an explicit `index_topk_pattern`. GLM-5.2's real pattern
(config `indexer_types`) is {0,1,2} + {2+4k} full, which the formula
mis-anchors — so derive the explicit pattern string ('F'/'S') from
indexer_types and let sglang's pattern branch take over.
"""

import logging
import os

logger = logging.getLogger(__name__)


def apply_fp8_gemm_backend_override():
    """Env-gated FP8 GEMM backend override (PARALLAX_FP8_GEMM_BACKEND=triton|cutlass|...).

    sglang 0.5.12's auto selection on SM120 (RTX 5090) picks FlashInfer CUTLASS groupwise
    FP8, whose scale-layout check rejects sglang's blockwise scales for non-128-multiple
    output dims — GLM-5.2-FP8's fused_qkv_a_proj is [2624, 6144] and dies at warmup with
    "expects B scale layout (k//block_k, n//block_n) ... got (21, 48); expected (48, 20)".
    Parallax builds ServerArgs itself and never plumbs fp8_gemm_runner_backend, so set the
    module global directly (get_fp8_gemm_runner_backend() falls back to it)."""
    backend = os.environ.get("PARALLAX_FP8_GEMM_BACKEND")
    if not backend:
        return
    try:
        from sglang.srt.layers.quantization import fp8_utils
        fp8_utils.FP8_GEMM_RUNNER_BACKEND = fp8_utils.Fp8GemmRunnerBackend(backend)
        logger.info("[fp8] forced FP8 GEMM runner backend = %s", backend)
    except Exception as e:  # noqa: BLE001 — never block startup on an optional override
        logger.warning("[fp8] could not force FP8 GEMM backend %r: %s", backend, e)


def apply_glm_moe_dsa_config_monkey_patch():
    try:
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (
            GlmMoeDsaConfig,
        )
    except ImportError:  # transformers without glm_moe_dsa: nothing to repair
        return

    amap = dict(getattr(GlmMoeDsaConfig, "attribute_map", {}))
    if amap.pop("head_dim", None) is not None:
        GlmMoeDsaConfig.attribute_map = amap
        logger.info(
            "[glm_moe_dsa] removed head_dim->qk_rope_head_dim attribute_map alias "
            "(config.json head_dim was clobbering qk_rope_head_dim)"
        )

    if getattr(GlmMoeDsaConfig.__init__, "_parallax_patched", False):
        return
    orig_init = GlmMoeDsaConfig.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        indexer_types = getattr(self, "indexer_types", None)
        if indexer_types and getattr(self, "index_topk_pattern", None) is None:
            self.index_topk_pattern = "".join(
                "F" if t == "full" else "S" for t in indexer_types
            )

    patched_init._parallax_patched = True
    GlmMoeDsaConfig.__init__ = patched_init
