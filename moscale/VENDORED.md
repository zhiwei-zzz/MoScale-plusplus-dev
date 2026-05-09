# MoScale (vendored)

This directory is a verbatim snapshot of the MoScale codebase, vendored as a
subdirectory of the MoScale-plusplus-dev repo for self-containment on the cluster.

- **Upstream:** https://github.com/zhiwei-zzz/MoScale
- **Pinned commit:** `c4b4d662aba366de71864dd3e20d10bcdf03d36c`
- **Snapshot taken:** 2026-05-09
- **Method:** `git archive HEAD | tar -x` (only tracked files at HEAD; no `.git`,
  no caches, no local artifacts).

## Relationship to the SkelVQ work in this repo

The parent repo (`MoScale-plusplus-dev`) is SALAD's codebase + our SkelVQ
extensions (a discrete tokenizer that replaces SALAD's Gaussian VAE bottleneck;
see `../SKELVQ_STATUS.md`). Stage 1 of the joint pipeline is the SkelVQ
tokenizer; stage 2 is MoScale's next-scale autoregressive transformer
(`moscale/model/transformer/moscale.py`).

Today the two halves are independent — SkelVQ trains on its own, MoScale trains
on its own. The integration (load a trained SkelVQ checkpoint as MoScale's
frozen first stage) is not yet implemented; that's the next milestone.

The MSQuantizer module from MoScale (`model/vq/quantizer.py`) was *also* copied
into the SALAD tree at `../models/vae/quantizer.py` so SkelVQ could use it as
one of three bottleneck options (`--quantizer_type msvq`). That copy is older
and predates this vendoring; once the integration lands we can dedupe and have
SkelVQ import directly from `moscale.model.vq.quantizer`.

## Updating

To bump to a new MoScale commit:

```bash
# from the SALAD repo root, with a sibling MoScale checkout at ../MoScale:
rm -rf moscale
mkdir -p moscale
git -C ../MoScale archive HEAD | tar -x -C moscale
# update VENDORED.md "Pinned commit" / "Snapshot taken" lines, then commit.
```
