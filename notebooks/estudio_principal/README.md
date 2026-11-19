# Estudio principal: código por ronda

El entrenamiento corre en Kaggle (2×T4); el análisis corre en CPU. Los `.ipynb` se generan desde los `.py` (`build_kaggle.py`); **la fuente es el `.py`**.
Qué resultado y qué archivo físico respalda cada cuaderno: [`EVIDENCIA.md`](../../EVIDENCIA.md).

Los scripts **no se movieron a subcarpetas** a propósito: comparten `a4m_lib.py`, las rutas relativas a `results/` y los kernels de Kaggle los importan por nombre. Mover archivos habría roto la reproducción.
La separación por ronda está en esta tabla y en el nombre de las carpetas de resultados (`raw/a` frente a `raw/a_T`, etc.).

## Código compartido (ambas rondas)

| Archivo | Qué hace |
|---|---|
| `a4m_lib.py` | Lectura de datos, particiones (`A4M_NBIG`, `A4M_GAP_S`, `A4M_SEG`), auditoría de fugas (`audit_splits`), entrenamiento, métricas (matrices de confusión por imagen) |
| `a4m_stats.py` | Bootstrap pareado por bloques de 10 imágenes, regla real / equivalente / inconcluso, corrección de Holm |
| `build_kaggle.py` | Genera los `.ipynb` desde los `.py` y los kernels de Kaggle |

## Ronda 1: 15,564 imágenes de entrenamiento, evaluación en M1–M3

| Orden | Archivo | Qué hace | Salida física (`results/estudio_principal/`) |
|---|---|---|---|
| 0 | `hipotesis_registradas.json`, `hipotesis_datos_completos.json` | Reglas de decisión escritas antes de correr (con fecha dentro del archivo) | (el propio archivo) |
| 1 | `A_datos_y_errores.py` | Teardown, P1 (apariencia: histograma, CLAHE, estandarización, desenfoque) y datos de P3 | `raw/a/` · `logs/A_teardown_P1_P3data.log` |
| 2 | `B_modelo.py` | P2: LoRA en dos bloques, zonas early / mid / late | `raw/b/` · `logs/B_P2_lora_*.log` |
| 3 | `D_baseline.py` | DeepLabV3-R50 con y sin preentrenamiento | `raw/d/` · `logs/D_baseline.log` |
| 4 | `C_resultados.py` | **Análisis en CPU**: tablas, figuras y veredictos de la ronda 1 | `P1_table.csv`, `P2_table.csv`, `baseline_table.csv`, `teardown_table.csv`, `veredictos.csv`, `posthoc_final_run.csv`, `figures/` |

`C_resultados.ipynb` es el único cuaderno guardado **con salidas** (se ejecutó). `A`, `B` y `D` están guardados sin salidas; su evidencia son los `raw/` y los `logs/`.

## Ronda 2: conjunto T (2,000 imágenes), 12,997 de entrenamiento

La ronda 2 reutiliza `A`, `B` y `D` con `A4M_NBIG` activado (hueco de 24 h y auditoría de fugas) y añade `E` y `F`.

| Orden | Archivo | Qué hace | Salida física (`results/estudio_principal/`) |
|---|---|---|---|
| 0 | `hipotesis_ronda2.json` | Reglas de la ronda 2, escritas el 6 nov 2026 antes de correr | (el propio archivo) |
| 1 | `A_datos_y_errores.py` | P1 sobre T | `raw/a_T/` (incluye `auditoria_datos.json`) |
| 2 | `B_modelo.py` | P2 sobre T (2 semillas por zona) | `raw/b_T/` |
| 3 | `D_baseline.py` | DeepLabV3 más largo sobre T | `raw/d_T/` |
| 4 | `E_ronda2.py` | Diseño 2×2 (decoder / LoRA × entropía cruzada / Lovász), 3 semillas | `raw/e/` · `R2_table.csv` · `figures_ronda2/` |
| 5 | `F_ronda2_resultados.py` | **Análisis en CPU**: checklist de fugas, veredictos de la ronda 2 y relectura post hoc de la ronda 1 | `veredictos_ronda2.csv`, `R2_vs_publicado.csv`, `sensibilidad_ronda1.csv` |
| 6 | `../../src/make_report_figures.py` | Curvas de entrenamiento y forest plot de los veredictos | `docs/informe_figuras/` |
| 7 | `../../figuras_poster/make_v3_figures.py` | Figuras 2–5 del póster | `figuras_poster/v3/` |

`E_ronda2.ipynb` está guardado **sin salidas**, y en el repo **no hay logs de consola de la ronda 2**. Su evidencia son los `raw/` (JSON con configuración, semilla y curva de validación, y NPZ con matrices de confusión por imagen).

## Cómo comprobar que todo coincide (CPU, sin dataset ni GPU)

```bash
pip install -r requirements.txt
A4M_RUN=full  python notebooks/estudio_principal/C_resultados.py      # ronda 1
python        notebooks/estudio_principal/F_ronda2_resultados.py      # ronda 2
python src/make_report_figures.py                                      # docs/informe_figuras/
cd figuras_poster && python make_v3_figures.py                         # figuras del póster
```

Estas cuatro órdenes se ejecutaron el 19 nov 2026: las tablas de ambas rondas se regeneraron con **el mismo contenido** que las versionadas y las cuatro figuras del póster salieron **idénticas byte a byte**.
Nota: `a4m_lib.py` importa `torch` aunque el análisis no entrene; sirve la versión CPU.
