# Estudio principal: índice del código (orden de ejecución)

Todo se entrena en Kaggle (2×T4). El análisis corre en CPU. Los `.ipynb` se generan desde los `.py` (`build_kaggle.py`); la fuente es el `.py`.

| Orden | Archivo | Qué hace | Ronda | Salida (`results/estudio_principal/raw/`) |
|---|---|---|---|---|
| 0 | `a4m_lib.py` | Lectura de datos, particiones (`A4M_NBIG`, `A4M_GAP_S`, `A4M_SEG`), auditoría de leakage, entrenamiento, métricas | ambas | — |
| 0 | `a4m_stats.py` | Bootstrap pareado por bloques, reglas real/equivalente/inconcluso, Holm | ambas | — |
| 0 | `hipotesis_registradas.json`, `hipotesis_datos_completos.json`, `hipotesis_ronda2.json` | Hipótesis y reglas escritas **antes** de correr (con fecha) | 1, 1 completa, 2 | — |
| 1 | `A_datos_y_errores.py` | Teardown, P1 (apariencia) y datos de P3 | 1 (`a`), 2 (`a_T`) | `P1_*.json/npz`, `teardown.json` |
| 2 | `B_modelo.py` | P2: LoRA por zona (early/mid/late) | 1 (`b`), 2 (`b_T`) | `P2_*.json/npz` |
| 3 | `D_baseline.py` | DeepLabV3-R50 con y sin ImageNet | 1 (`d`), 2 (`d_T`) | `BL_*.json/npz` |
| 4 | `E_ronda2.py` | Diseño 2×2 (decoder/LoRA × CE/Lovász), 16 épocas, 3 semillas | 2 (`e`) | `R2_*.json/npz` |
| 5 | `C_resultados.py` | Tablas y figuras de la ronda 1 | 1 | `results/estudio_principal/*.csv` |
| 6 | `F_ronda2_resultados.py` | Checklist de leakage + veredictos de la ronda 2 + relectura post hoc de la ronda 1 | 2 | `veredictos_ronda2.csv`, `R2_vs_publicado.csv`, `sensibilidad_ronda1.csv` |
| 7 | `../../src/make_report_figures.py` | Curvas de entrenamiento y forest plot de veredictos | 2 | `docs/informe_figuras/` |

Notas: (1) el mapa completo con kernels de Kaggle, datasets y entorno está en `docs/INFORME_PARA_POSTER.md` §9. (2) `E_ronda2.ipynb` es el cuaderno ejecutado de E. (3) Los scripts quedaron sin reestructurar a propósito, para no romper la tubería de Kaggle antes de la entrega.
