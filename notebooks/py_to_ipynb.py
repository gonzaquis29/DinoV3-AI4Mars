"""Convert a `# %%`-cell .py file into a .ipynb (cells split at lines starting with `# %%`)."""
import json, sys
from pathlib import Path

src = Path(sys.argv[1]); cells, cur, kind = [], [], None


def flush():
    if kind is None or not "".join(cur).strip():
        return
    text = "".join(cur).rstrip("\n")
    if kind == "markdown":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n"))
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    cells.append(c)


for line in src.read_text().splitlines(True):
    if line.startswith("# %%"):
        flush(); cur = []
        kind = "markdown" if "[markdown]" in line else "code"
        if kind == "code":
            cur.append(line)            # keep the cell title as a comment
    else:
        cur.append(line)
flush()
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = src.with_suffix(".ipynb"); out.write_text(json.dumps(nb, indent=1)); print(out, len(cells), "cells")
