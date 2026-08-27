# data/

`final/` is **tracked** — the 330-descriptor feature matrices for all three
splits ship with this repository as Parquet and are ready to use. Everything
else here is bulk or regenerable and is not tracked.

| Path | Tracked | Contents |
|---|---|---|
| `final/*.parquet` | yes | 330-descriptor matrices: fold1 (32,948), fold2 (32,922), test (16,489) |
| `final/CHECKSUMS.json` | yes | SHA-256, shapes and label counts for each file |
| `final_v2/` | no | same matrices with the corrected CTD block — `scripts/regen_ctd_fixed.py` |
| `preds_prob/` | no | probability cache written by `scripts/evaluate.py` |
| `escape_repro/` | no | released ESCAPE checkpoints and structural maps |

See [`../docs/DATA.md`](../docs/DATA.md) for the schema, the descriptor
definitions, why the matrices are Parquet rather than CSV, and how to obtain the
ESCAPE checkpoints.
