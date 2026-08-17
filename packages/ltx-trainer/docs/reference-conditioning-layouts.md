# Reference conditioning: RoPE layouts and source phase

`FlexibleStrategy`'s `reference` condition concatenates clean reference latents to the target
sequence (the IC-LoRA recipe). By default the reference reuses the target's RoPE coordinates,
so the model can only tell source from target implicitly — clean vs noisy tokens, plus
concatenation order.

This page documents two optional mechanisms that make that distinction explicit:

- **`layout`** — *where* the reference sits in the RoPE coordinate grid.
- **`source_phase`** — an independent rotary tag per source, composed with the positional RoPE.

Both are **off by default**. A config that does not opt in produces byte-identical behaviour to
upstream: `layout="overlap"` returns the reference positions unchanged, and a `segment_ids`
field that is all-zero (or `None`) is an exact no-op in the rotary maths.

## Background

The `st_drc` layout implements the coordinate separation described in **ST-DRC**
([arXiv:2606.02441](https://arxiv.org/abs/2606.02441)): the reference block is shifted past the
target's extent on every axis, so source and target occupy disjoint regions of the coordinate
space and cannot collide positionally.

**`overlap` + `source_phase` is not from that paper — it is an addition made here.** Instead of
moving the reference away from the target, it keeps the reference on the *same* coordinates and
separates the sources with an independent rotary phase. In practice this has trained
noticeably better than the literal coordinate shift, across several weeks of runs on identity
transfer, head-swap and Face-ID style tasks. It has been used on **both LTX-2.3 and LTX 2.5** —
the mechanism is positional, so it does not depend on the text-encoder or feature-extractor
differences between versions.

**Recommended default: `layout: overlap` with `source_phase: true`.** Start there. The other
layouts are available for cases where you specifically want disjoint coordinates.

### Released example

[**Alissonerdx/LTX-Best-Face-ID**](https://huggingface.co/Alissonerdx/LTX-Best-Face-ID) is a
public identity-transfer LoRA for LTX-2.3 trained with this recipe — `layout: overlap`,
`source_phase: true`, `source_id: 2`. The repository includes the LoRA, the character-sheet
variant, the reference sheets used, and the prompt format, so the configuration below can be
checked against a working artefact rather than taken on faith.

## Layouts

| `layout` | What it does | Notes |
|---|---|---|
| `overlap` | Reference reuses the target's coordinates. | Upstream behaviour. **Recommended**, combined with `source_phase`. |
| `st_drc` | Shifts the reference past the target's extent on **every** axis (T, H, W). | The literal ST-DRC construction. |
| `sidecar` | Parks the reference as a panel to the right of the target, vertically centred, spanning the clip temporally. | `virtual_sidecar` is an accepted alias. `sidecar_margin_pixels` sets the gap. |
| `strata` | Shifts **only** the temporal axis to an absolute band; H/W keep overlapping the target. | For stacked memory-style references. Requires `strata_slot` (`ltm` or `stm`). |

`strata` follows the band construction from Strata-RoPE: memory slots sit on fixed absolute
temporal bands near `strata_f_lim`, far from the current shot, so the rotary attenuates
cross-band interaction by distance. The paper's `f_lim=128` is a large distance; on LTX's short
latent-frame counts that can attenuate a reference to near-zero influence, which is why
`strata_f_lim` is configurable — lower it to keep the band close enough to still condition.

## Source phase

With `source_phase: true`, reference tokens receive a rotary phase of `source_id * phase_scale`,
composed with the ordinary spatiotemporal RoPE. Target tokens always carry source `0`, which is
an exact no-op, so the target's positional encoding is untouched.

The phase is norm-preserving (it rotates the RoPE frequencies rather than scaling them), and it
is independent of the positional axes — so a reference can share the target's exact coordinates
and still be unambiguously identifiable.

```yaml
conditions:
  - type: reference
    latents_dir: reference_latents
    layout: overlap        # keep the target's coordinates
    source_phase: true     # separate the sources by rotary phase instead
    source_id: 2           # target is always 0
    phase_scale: 1.0
```

### Multiple references

Distinct `source_id` values give distinct phases, so several references can share the target's
coordinates and remain separable — a second reference at `source_id: 3`, a third at `4`, and so
on. Configure one `reference` condition per source:

```yaml
conditions:
  - type: reference
    latents_dir: reference_a_latents
    layout: overlap
    source_phase: true
    source_id: 2
  - type: reference
    latents_dir: reference_b_latents
    layout: overlap
    source_phase: true
    source_id: 3
```

> **Validated scope.** What has actually been trained and verified so far is the **single
> reference at `source_id: 2`** case. Multiple simultaneous sources at distinct ids are
> supported by the implementation and are a natural extension, but they have not been validated
> here — treat them as untested.

## Keeping training and inference aligned

The layout and phase are geometry, not weights. A checkpoint trained with a given layout must be
sampled with the **same** layout, or the model sees a coordinate frame it never saw in training.

Three call sites share one implementation
(`ltx_core.conditioning.reference_layout`) so they cannot drift:

| Path | Where |
|---|---|
| Training | `FlexibleStrategy._apply_reference_condition` |
| Validation during training | `ValidationRunner`, via the same fields on the validation condition |
| Inference | `VideoConditionByReferenceLatent` (ltx-core) |

Validation conditions therefore mirror the training fields:

```yaml
validation:
  samples:
    - prompt: "..."
      conditions:
        - type: reference
          video: /path/to/reference.mp4
          layout: overlap
          source_phase: true
          source_id: 2
```

## ComfyUI

`VideoConditionByReferenceLatent` carries the full parameter set, so any pipeline built on
ltx-core inherits the behaviour. There are currently **no official LTX ComfyUI nodes exposing
`layout` / `source_phase`**, so a LoRA trained with these options needs a node that sets them —
today that means the community BFS nodes (`ltx_identity_overlap`), which reimplement the same
geometry. Adding official nodes would remove that dependency, and is the natural follow-up to
this PR.

### Reference sizing

One detail that matters in practice and has no training-side equivalent: how the reference image
is resized before encoding. The BFS node exposes three modes, and the right one depends on what
the checkpoint was trained with:

| Mode | Behaviour | When to use |
|---|---|---|
| `match_target` | Centre-crop then resize the reference to the output's pixel size. | Reference resolution never mattered (e.g. single face crops). Discards whatever is outside the centre when aspect ratios differ. |
| `match_target_letterbox` | Same target pixel size, but fits the whole reference inside it, preserving aspect ratio and padding. | Mismatched aspect ratios where nothing may be cut — e.g. a landscape composite sheet driving a portrait output. |
| `native_resolution` | Encode the reference at its own size, rounded to a multiple of 32, independent of the output. | Checkpoints trained on a **fixed reference resolution bucket** that differs from the output's bucket. |

If a checkpoint was trained with references in their own fixed bucket, sampling it with
`match_target` silently changes the reference's token count and grid extent, and identity
transfer degrades. Match the training-time preprocessing.

The BFS node also accepts a *batch* of reference images, assigning `source_id + index` to each —
the inference-side counterpart of the multi-reference configuration above (and likewise
untested here beyond the single-reference case).

## Backwards compatibility

- `layout` defaults to `overlap` and `source_phase` to `false`, so existing configs are unchanged.
- `apply_reference_layout` returns the input tensor for `overlap`.
- `extend_segment_ids` returns `None` when there is nothing to tag, preserving the exact
  upstream fast path through the rotary code.
- Source `0` is an algebraic no-op in `apply_segment_phase_to_freqs_cis`.

## Config reference

| Field | Default | Meaning |
|---|---|---|
| `layout` | `overlap` | `overlap`, `st_drc`, `sidecar` (alias `virtual_sidecar`), `strata` |
| `source_phase` | `false` | Enable the per-source rotary tag |
| `source_id` | `2` | Source index for this reference; target is always `0` |
| `phase_scale` | `1.0` | Multiplier on the source phase |
| `sidecar_margin_pixels` | `0.0` | Gap between target and sidecar panel |
| `strata_slot` | `null` | `ltm` or `stm`; required by `layout: strata` |
| `strata_f_lim` | `128` | Ceiling of the absolute Strata-RoPE bands |

Video references only — the fields are rejected on an audio reference.
