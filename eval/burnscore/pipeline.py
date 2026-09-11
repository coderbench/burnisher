"""Compose stages into the end-to-end generation, and say what share each one holds.

The dominance question -- "is what a contributor can touch at least 20% of total wall time?" --
cannot be asked of a stage in isolation, because the answer is set by how often the stage runs
rather than by how big it is. PixArt-Sigma's text encoder holds 89% of the checkpoint's
parameters and runs ONCE; its DiT holds 11% and runs `2 x steps` times under classifier-free
guidance. Any screen that ranked stages by parameter count would send a contributor to spend a
week on 2% of the clock.

So invocation counts are first-class here, and they are the thing that changes between
generations: at four steps instead of twenty the same three stages reorder completely. That is
not a flaw in the screen, it is the REGENERATION property the screen is looking for.
"""
from __future__ import annotations

from . import geometry as G
from .roofline import bound_for


def pixart_stages(candidate: dict, *, resolution=1024, steps=20, caption_len=None,
                  cfg=True, wdtype="bf16", adtype="bf16", attn_impl="flash",
                  vae_attn_impl="flash"):
    """The three stages of a PixArt-Sigma generation, with how often each runs.

    `cfg` doubles the DiT batch rather than doubling the invocation count, because that is what
    the runtime does -- one forward pass over a batch of two. The distinction matters for the
    roofline: a batch of two reads the weights once, two sequential passes read them twice.
    """
    cap = caption_len or int(candidate["text_encoder"]["max_sequence_length"])
    batch = 2 if cfg else 1
    text = G.t5_encoder(candidate["text_encoder"], seq=cap, batch=batch,
                        wdtype=wdtype, adtype=adtype, attn_impl=attn_impl)
    text.invocations = 1
    dit = G.pixart_dit(candidate["denoiser"], resolution=resolution, caption_len=cap,
                       batch=batch, wdtype=wdtype, adtype=adtype, attn_impl=attn_impl)
    dit.invocations = steps
    vae = G.vae_decoder(candidate["vae"], resolution=resolution, batch=1,
                        wdtype=wdtype, adtype=adtype, attn_impl=vae_attn_impl)
    vae.invocations = 1
    vae.notes.append("batch 1: CFG is resolved into a single latent before the decoder runs.")
    return [text, dit, vae]


def shares(stages, device, *, device_name=None, use="ceiling"):
    """Each stage's share of the pipeline, by the chosen bound.

    `use="ceiling"` answers "where would the time go on a perfect implementation" and
    `use="decomposed"` answers "where does it go on one built the obvious way". They disagree,
    and the disagreement is itself informative: a stage whose share is much larger decomposed
    than ideal is a stage whose time is in intermediate traffic, which is the fusion surface.
    """
    if use not in ("ceiling", "decomposed"):
        raise ValueError("use must be 'ceiling' or 'decomposed'")
    rows = []
    for s in stages:
        b = bound_for(s, device, cell=s.stage, device_name=device_name)
        secs = b.ceiling_seconds if use == "ceiling" else b.decomposed_seconds
        rows.append({"stage": s.stage, "invocations": s.invocations, "bound": b,
                     "seconds": secs, "flops": s.flops,
                     "unavoidable_bytes": s.unavoidable_bytes,
                     "param_bytes": s.param_bytes, "bound_by": b.bound_by})
    total = sum(r["seconds"] for r in rows)
    for r in rows:
        r["share"] = (r["seconds"] / total) if total else 0.0
    return {"rows": rows, "total_seconds": total, "basis": "model", "bound_used": use,
            "_basis_note": "Arithmetic ceilings, not measurements. Shares computed from them "
                           "are predictions about where time WOULD go, and the ordering is "
                           "more trustworthy than the magnitudes."}


def resident_bytes(stages, *, resident_all=True):
    """Peak parameter residency for the pipeline, which is the memory objective's base.

    `resident_all=True` is the naive arrangement: every stage's weights live on the device for
    the whole generation. It is what a first implementation does and it is why the 32 GB card is
    the binding constraint -- the T5 encoder alone is 9.5 GB in bf16 and is dead weight for the
    entire denoise loop. `resident_all=False` is the streaming arrangement, where the peak is
    the largest single stage. The difference between the two numbers IS the offload backlog item.
    """
    per = {s.stage: s.param_bytes for s in stages}
    return {"per_stage": per,
            "peak_bytes": sum(per.values()) if resident_all else max(per.values()),
            "arrangement": "all-resident" if resident_all else "streamed",
            "_note": "Parameters only. Activations, the CUDA context and the allocator's slack "
                     "are real and are not counted here; the memory objective is measured from "
                     "the device, not from this number."}
