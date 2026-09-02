# Third-Party Code Attribution

This document records the third-party components this project redistributes in source
form, the third-party model weights the pipeline requires, and the third-party code it
fetches at run time. It is a **summary for convenience**: the licence texts themselves
govern. Copyright notices are preserved in `LICENSE`, `NOTICE`, and `pixal3d/LICENSE`.

---

## 1. Pixal3D — generative backbone (redistributed in this repository)

**Directory:** `pixal3d/`
**Source:** TencentARC
**Repository:** https://github.com/TencentARC/Pixal3D
**Licence:** MIT
**Usage:** The single-image 3D generation backbone this work builds on. Its DiT blocks
are used frozen; the multi-view conditioning and LoRA adapters in `pixal3d_multiview/`
are added around it.

Copyright notice (reproduced verbatim in `pixal3d/LICENSE` and in `LICENSE` Part 2):

```
MIT License

Copyright (c) 2026 Tencent.
```

The vendored copy is adapted where the inference pipeline required it. MIT permits
modification and redistribution provided the copyright notice and permission notice are
retained; any such changes are covered by the same MIT terms and are **not** subject to
this project's Apache-2.0 grant.

**Components declared in Pixal3D's own `NOTICE`.** Because the backbone is vendored,
these upstream attributions carry over:

| Component | Copyright | Licence |
|---|---|---|
| TRELLIS.2 | Microsoft Corporation | MIT |
| dinov2 | the dinov2 authors | Apache-2.0 |
| Direct3D-S2 | DreamTech | MIT |
| MoGe | Microsoft Corporation | MIT |

---

## 2. TRELLIS.2 — environment prerequisite

**Source:** Microsoft
**Repository:** https://github.com/microsoft/TRELLIS.2
**Licence:** MIT
**Usage:** Pixal3D is built on TRELLIS.2, and installation Step 1 directs users to
TRELLIS.2's guide to set up the base environment. TRELLIS.2 is **not** separately
vendored in this repository.

---

## 3. Model weights — downloaded by the user, not redistributed

None of the weights below are contained in this repository. Each is fetched by the user
and remains subject to its own terms; this project grants no rights in any of them.

| Weights | Origin | Licence | Note |
|---|---|---|---|
| Pixal3D base checkpoints | TencentARC — https://huggingface.co/TencentARC/Pixal3D | MIT | Fetched in installation Step 3 |
| DINOv3 ViT-L/16 | Meta — https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m | **DINOv3 Licence** | **Neither Apache-2.0 nor MIT**, and access is **gated**: the user must accept Meta's terms before download. Used as a frozen feature extractor |
| BiRefNet | https://huggingface.co/ZhengPeng7/BiRefNet | its own upstream licence | Reached only through the upstream single-image path, which the released scene pipeline does not use |

The finetuned LoRA adapters and multi-view aggregator released with this project are our
own work and are licensed as stated in `LICENSE`. They are deltas on top of the Pixal3D
base weights and are not usable without separately obtaining those.

---

## 4. Code fetched and executed at run time

Parts of the upstream pipeline download third-party code over the network on first use
and execute it. That code is not contained in this repository and remains under its own
licence. Users behind restricted networks, or subject to policies on executing remotely
fetched code, should be aware of it:

| Fetched from | Mechanism | Where | Purpose |
|---|---|---|---|
| `valeoai/NAF` | `torch.hub` | `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py` | upsampler used by the conditioning path |
| `facebookresearch/dinov2` | `torch.hub` | `pixal3d/modules/image_feature_extractor.py`, `pixal3d/trainers/flow_matching/mixins/image_conditioned.py` | alternative image encoder |
| `ZhengPeng7/BiRefNet` | Hugging Face, `trust_remote_code=True` | `pixal3d/pipelines/rembg/BiRefNet.py` | background removal on the upstream single-image path |
