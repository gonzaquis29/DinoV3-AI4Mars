"""Builds the Kaggle folders for the v2 run (nothing is pushed here).

    python notebooks/v2/build_kaggle.py            # -> kaggle_run/v2/{lib, a-quick, b-quick, a, b}

lib      : private dataset <your-kaggle-user>/a4m-v2-lib with a4m_lib.py and hypotheses_v2.json
a / b    : notebook kernels (the .ipynb generated from A_datos_y_errores.py / B_modelo.py)
*-quick  : the same notebooks with one extra first cell that sets A4M_QUICK=1 (64 images, 1 epoch)

Push:  kaggle datasets create -p kaggle_run/v2/lib   (first time; then: kaggle datasets version -p ... -m msg)
       kaggle kernels push -p kaggle_run/v2/a-quick --accelerator NvidiaTeslaT4 -t 3600
"""
import json, os, shutil, subprocess, sys
from pathlib import Path

USER = os.environ.get("KAGGLE_USER", "your-kaggle-user")      # your Kaggle username
here = Path(__file__).resolve().parent
out = here.parents[1] / "kaggle_run" / "v2"
DATA = ["darshannnnnn/ai4mars-raw", f"{USER}/dinov3-vitb16-lvd1689m", f"{USER}/a4m-v2-lib"]

lib = out / "lib"; lib.mkdir(parents=True, exist_ok=True)
for f in ("a4m_lib.py", "hypotheses_v2.json"):
    shutil.copy(here / f, lib / f)
json.dump({"title": "a4m-v2-lib", "id": f"{USER}/a4m-v2-lib", "licenses": [{"name": "other"}]}, open(lib / "dataset-metadata.json", "w"), indent=1)

for src, slug in (("A_datos_y_errores.py", "a"), ("B_modelo.py", "b"), ("D_baseline.py", "d")):
    subprocess.run([sys.executable, str(here.parent / "py_to_ipynb.py"), str(here / src)], check=True)
    nb = json.load(open(here / src.replace(".py", ".ipynb")))
    for mode in ("", "quick", "full"):
        if mode == "full" and slug == "d":
            continue
        quick = mode == "quick"
        d = out / (f"{slug}-{mode}" if mode else slug); d.mkdir(exist_ok=True)
        cells = list(nb["cells"])
        if mode == "full":
            cells.insert(0, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                             "source": ["import os; os.environ['A4M_FULL'] = '1'   # final version: all training images, early stopping\n"]})
        if quick:
            cells.insert(0, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                             "source": ["import os; os.environ['A4M_QUICK'] = '1'   # quick check: 64 images, 1 epoch\n"]})
        json.dump({**nb, "cells": cells}, open(d / "notebook.ipynb", "w"), indent=1)
        name = f"ai4mars-v2-{slug}" + (f"-{mode}" if mode else "")
        json.dump({"id": f"{USER}/{name}", "title": name.replace("-", " "), "code_file": "notebook.ipynb", "language": "python",
                   "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                   "dataset_sources": DATA + ([f"{USER}/a4m-v2-a-partial"] if slug == "a" and mode == "" else []), "competition_sources": [], "kernel_sources": [], "model_sources": [],
                   "machine_shape": "NvidiaTeslaT4"}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", out, sorted(p.name for p in out.iterdir()))
