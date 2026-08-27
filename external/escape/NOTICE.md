# Third-party code: ESCAPE

The files in this directory are **not** part of this work. They are the model
and dataset definitions of the ESCAPE benchmark, reproduced here so that the
released ESCAPE checkpoints can be loaded and re-run for the paired comparison
in Section 3.2 and the structural-branch ablation in Section 3.4 of the paper.

| File | Origin |
|---|---|
| `models.py` | ESCAPE — `ClassifierTransformer` and the multimodal head |
| `dataset.py` | ESCAPE — sequence tokenisation and structural-map loading |

Upstream source: <https://github.com/BCV-Uniandes/ESCAPE>
Benchmark data: <https://doi.org/10.7910/DVN/C69MCD>

Please cite the ESCAPE publication if you use this code, and refer to the
upstream repository for its licence terms. The MIT licence at the root of this
repository covers our own code and does **not** extend to this directory.
