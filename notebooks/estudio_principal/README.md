# Código del estudio

El entrenamiento corre en Kaggle (2×T4); el análisis corre en CPU. Los `.ipynb` se generan desde los `.py` con `build_kaggle.py` (**la fuente es el `.py`**).

| Archivo | Qué hace | Salida (`results/estudio_principal/`) |
|---|---|---|
| `a4m_lib.py`, `a4m_stats.py` | Datos, particiones, entrenamiento, métricas; bootstrap pareado y reglas de decisión | — |
| `A_datos_y_errores` | Teardown, P1 (apariencia), datos de P3 | `raw/a`, `raw/a_T` |
| `B_modelo` | P2: LoRA por profundidad (early / mid / late) | `raw/b`, `raw/b_T` |
| `D_baseline` | DeepLabV3-R50 con y sin preentrenamiento | `raw/d`, `raw/d_T` |
| `C_resultados` | Análisis ronda 1 (CPU): tablas, figuras, veredictos | `*_table.csv`, `veredictos.csv`, `figures/` |
| `E_ronda2` | Diseño 2×2 (decoder / LoRA × CE / Lovász) | `raw/e`, `R2_table.csv`, `figures_ronda2/` |
| `F_ronda2_resultados.py` | Análisis ronda 2 (CPU): veredictos | `veredictos_ronda2.csv` |
| `hipotesis_*.json` | Reglas de decisión escritas antes de correr | — |

Sufijo `_T`: ronda 2 (conjunto T). `a4m_lib.py` importa `torch` aunque el análisis no entrene; sirve la versión CPU.
