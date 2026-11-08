"""Builds the Kaggle folders for the main run (nothing is pushed here).

    python notebooks/estudio_principal/build_kaggle.py            # -> kaggle_run/v2/{lib, a, b, d, e, *-quick, *-full, b-long}

lib      : private dataset <your-kaggle-user>/a4m-v2-lib with a4m_lib.py and the registered hypotheses (round 1 and round 2)
a / b    : notebook kernels (the .ipynb generated from A_datos_y_errores.py / B_modelo.py)
e        : round 2 (E_ronda2.py): decoder / LoRA / Lovasz 2x2, E_EPOCHS epochs (default 20), 3 seeds, with the held-out block T. Full data.
a-t / b-t / d-t : the round-1 notebooks A (appearance), B (where to adapt, B_EPOCHS epochs, default 8, results prefixed P2T) and D (baseline,
           150 min budget) re-run with the held-out block T (A4M_NBIG=2000), so that the hypotheses are decidable
           Epochs: E_EPOCHS=24 B_EPOCHS=10 python build_kaggle.py
*-quick  : the same notebooks with one extra first cell that sets A4M_QUICK=1 (64 images, 1 epoch): run it first, it costs minutes

Push:  kaggle datasets create -p kaggle_run/v2/lib   (first time; then: kaggle datasets version -p kaggle_run/v2/lib -m msg)
       kaggle kernels push -p kaggle_run/v2/e-quick --accelerator NvidiaTeslaT4 -t 3600     # check first
       kaggle kernels push -p kaggle_run/v2/e       --accelerator NvidiaTeslaT4 -t 43200    # 12 h session limit
Resume after a cut: attach the previous kernel output as an input dataset; finished seeds are skipped, a half-done seed resumes from its epoch checkpoint.
"""
import json, os, shutil, subprocess, sys
from pathlib import Path

USER = os.environ.get("KAGGLE_USER", "your-kaggle-user")      # your Kaggle username
here = Path(__file__).resolve().parent
out = here.parents[1] / "kaggle_run" / "v2"
DATA = ["darshannnnnn/ai4mars-raw", f"{USER}/dinov3-vitb16-lvd1689m", f"{USER}/a4m-v2-lib"]

lib = out / "lib"; lib.mkdir(parents=True, exist_ok=True)
for f in ("a4m_lib.py", "a4m_stats.py", "hipotesis_registradas.json", "hipotesis_ronda2.json"):
    shutil.copy(here / f, lib / f)
json.dump({"title": "a4m-v2-lib", "id": f"{USER}/a4m-v2-lib", "licenses": [{"name": "other"}]}, open(lib / "dataset-metadata.json", "w"), indent=1)


def setenv(**kv):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": ["import os\n"] + [f"os.environ[{k!r}] = {v!r}\n" for k, v in kv.items()]}


# slug -> (source, modes). "" = the notebook as is. E has no "full" mode: it always trains on all images.
E_EPOCHS, B_EPOCHS, A_EPOCHS = os.environ.get("E_EPOCHS", "16"), os.environ.get("B_EPOCHS", "15"), os.environ.get("A_EPOCHS", "15")
JOBS = {"a": ("A_datos_y_errores.py", ("", "quick", "full", "t")), "b": ("B_modelo.py", ("", "quick", "full", "t")),
        "d": ("D_baseline.py", ("", "quick", "t")), "e": ("E_ronda2.py", ("", "quick", "run"))}
for slug, (src, modes) in JOBS.items():
    subprocess.run([sys.executable, str(here.parent / "py_to_ipynb.py"), str(here / src)], check=True)
    nb = json.load(open(here / src.replace(".py", ".ipynb")))
    for mode in modes:
        d = out / (f"{slug}-{mode}" if mode else slug); d.mkdir(exist_ok=True)
        cells = list(nb["cells"])
        if mode == "full":
            cells.insert(0, setenv(A4M_FULL="1"))                    # final version: all training images, early stopping
        if mode == "t":                                               # round 2: with the held-out block T so that hypotheses are decidable
            env = {"a": dict(A4M_FULL="1", A4M_EPOCHS=A_EPOCHS, A4M_PATIENCE="0"),      # probes: full schedule, best epoch by validation (no early stopping)
                   "b": dict(A4M_FULL="1", A4M_EPOCHS=B_EPOCHS, A4M_ZONES="late,early,mid", A4M_SEEDS="0,1", A4M_P2_TAG="P2T"),   # late and early first: they are the key contrast
                   "d": dict(BUDGET_MIN="360")}[slug]                                           # baseline: patience 3 on validation, capped at 6 h
            cells.insert(0, setenv(A4M_NBIG="2000", **env))
        if mode == "run":                                             # E: the real run (epochs fixed at build time)
            cells.insert(0, setenv(A4M_EPOCHS=E_EPOCHS))
        if mode == "quick":
            cells.insert(0, setenv(A4M_QUICK="1"))                    # quick check: 64 images, 1 epoch
        json.dump({**nb, "cells": cells}, open(d / "notebook.ipynb", "w"), indent=1)
        name = f"ai4mars-v2-{slug}" + (f"-{mode}" if mode else "")
        json.dump({"id": f"{USER}/{name}", "title": name.replace("-", " "), "code_file": "notebook.ipynb", "language": "python",
                   "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                   "dataset_sources": DATA + ([f"{USER}/a4m-v2-a-partial"] if slug == "a" and mode == "" else []), "competition_sources": [], "kernel_sources": [], "model_sources": [],
                   "machine_shape": "NvidiaTeslaT4"}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", out, sorted(p.name for p in out.iterdir()))
