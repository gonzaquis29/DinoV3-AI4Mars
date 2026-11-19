# Evidencia: qué resultado sale de qué cuaderno y dónde está el archivo

Mapa para comprobar el póster (v4) y el artículo. Cada fila apunta a un archivo físico del repositorio.
El código está organizado por ronda en [`notebooks/estudio_principal/README.md`](notebooks/estudio_principal/README.md).

- **Ronda 1:** 15,564 imágenes de entrenamiento, evaluación en los tests de expertos M1–M3. Carpetas `raw/a`, `raw/b`, `raw/d`.
- **Ronda 2:** conjunto interno T (2,000 imágenes, etiquetas de voluntarios, **exploratorio**), 12,997 de entrenamiento. Carpetas `raw/a_T`, `raw/b_T`, `raw/d_T`, `raw/e`.
- **Estudio por capas (anterior):** `notebooks/ai4mars_dinov3_final.ipynb` → `results/final_run/`.

Todas las rutas de `raw/`, tablas y figuras son relativas a `results/estudio_principal/` salvo que se indique otra cosa.
Cada corrida deja un `.json` (configuración, semilla, mejor época, curva de validación) y un `.npz` (matrices de confusión 4×4 **por imagen**, de las que se recalcula cualquier cifra).

## 1. Por cuaderno

| Cuaderno | Ronda | Qué produce | Evidencia física | Respalda |
|---|---|---|---|---|
| `A_datos_y_errores` | 1 | Teardown, P1 apariencia, datos de P3 | `raw/a/` (31 archivos) · `logs/A_teardown_P1_P3data.log` | Teardown (86 M parámetros, 215 GFLOPs); H1 en M1/M2; datos de H3 |
| `A_datos_y_errores` | 2 | P1 sobre T | `raw/a_T/` (32 archivos, con `auditoria_datos.json`) | H1 en T (CLAHE −0.010) |
| `B_modelo` | 1 | P2 LoRA early / mid / late | `raw/b/` (15 archivos) · `logs/B_P2_lora_*.log` | H2 en M1/M2 (inconclusa) |
| `B_modelo` | 2 | P2 sobre T, 2 semillas por zona | `raw/b_T/` (16 archivos) | H2 en T (+0.033) |
| `D_baseline` | 1 | DeepLabV3-R50 con y sin ImageNet, 3 épocas | `raw/d/` (4 archivos) · `logs/D_baseline.log` | Comparación de la tabla del README |
| `D_baseline` | 2 | DeepLabV3 más largo sobre T | `raw/d_T/` (5 archivos) | C3 y la barra DeepLabV3 de la Fig. 2 |
| `C_resultados` (CPU) | 1 | Tablas, figuras y veredictos | `P1_table.csv`, `P2_table.csv`, `baseline_table.csv`, `teardown_table.csv`, `veredictos.csv`, `posthoc_final_run.csv`, `figures/` | Veredictos HP1–HP3b; 0.862 / 0.546; ECE 0.029; brecha 0.674 |
| `E_ronda2` | 2 | Diseño 2×2 decoder/LoRA × CE/Lovász | `raw/e/` (25 archivos) · `R2_table.csv` · `figures_ronda2/` | Mejor modelo 0.682 / 0.732 / 0.867; C1; C2 |
| `F_ronda2_resultados` (CPU) | 2 | Veredictos de la ronda 2 | `veredictos_ronda2.csv`, `R2_vs_publicado.csv`, `sensibilidad_ronda1.csv` | H1, H2, C1–C4 del póster |
| `ai4mars_dinov3_final` | anterior | Sonda por capa, decoder, ajuste fino, controles | `results/final_run/` (`table_main.csv`, `hypothesis_verdicts.csv`, `results/`) | Tabla del estudio por capas del README; DINOv3 frente a DINOv2; ViT aleatorio |

## 2. Cada cifra del póster

| Afirmación del póster | Dónde está la evidencia |
|---|---|
| Sonda 0.674 / 0.723 / 0.862 (4,612 parámetros) | Ronda 1: `raw/a/P1_original_s{0,1,2}.npz` → `baseline_table.csv`, `P1_table.csv` |
| Decoder + Lovász 0.682 / 0.732 / 0.867 | `raw/e/R2_dec_lov_s{0,1,2}.npz` → `R2_table.csv` |
| DeepLabV3 0.675 / 0.727 / 0.884 | `raw/d_T/BL_pre_full_s0.npz` (se recalcula de la matriz; no hay un CSV dedicado). La tabla `baseline_table.csv` es **otra** corrida (ronda 1, 3 épocas): 0.667 / 0.713 / 0.815 |
| Valores publicados (SegFormer-B3, UPerNet-Swin-S) | `results/final_run/table_published.csv` (Mohammad, Li y Gao, iSpaRo 2025); comparación en `R2_vs_publicado.csv` |
| H1: CLAHE −0.010 real; estandarización y desenfoque equivalentes | `veredictos_ronda2.csv` (HRA, HRAb) ← `raw/a_T/`; en M1 `P1_table.csv` y `sensibilidad_ronda1.csv` |
| H2: late − early +0.033 en T; +0.009 en M1 sin confirmar | `veredictos_ronda2.csv` (HRW) ← `raw/b_T/`; ronda 1 en `P2_table.csv` y `veredictos.csv` (HP2) |
| C1: Lovász +0.024 en IoU de roca grande, más falsos positivos | `veredictos_ronda2.csv` (HR1) ← `raw/e/` |
| C2: LoRA sobre el decoder +0.006, inconcluso | `veredictos_ronda2.csv` (HR2) |
| C3: sonda frente a DeepLabV3 −0.005, inconcluso | `veredictos_ronda2.csv` (HRB) ← `raw/d_T/` |
| C4: empate con los publicados | `R2_vs_publicado.csv` |
| H3: 0.862 frente a 0.546, +0.315 [+0.180, +0.413] | `veredictos.csv` (HP3a) ← `raw/a/` |
| ECE 0.029; brecha de confianza 0.674 en roca grande | `veredictos.csv` (HP3b); figura `figures/P3_calibracion.png`. Es la **sonda en el test de expertos** (ronda 1), no la Fig. 4 |
| Calibración en T por cabeza (Fig. 4) | `figuras_poster/v3/figD_calibration.png` ← `raw/e/` (también `figures_ronda2/6e_calibracion.png`) |
| IoU por clase en T y falsos positivos en M3 (Fig. 5) | `figuras_poster/v3/figC_perclass.png` ← `raw/e/` (también `figures_ronda2/6c_iou_clase.png`) |
| Hueco de 24 h y auditoría de fugas (ronda 2) | `raw/{a_T,b_T,d_T,e}/auditoria_datos.json`; código `audit_splits` en `a4m_lib.py` |
| Regla escrita antes de correr | `notebooks/estudio_principal/hipotesis_*.json` (la fecha está **dentro** del archivo; git solo la tiene desde el commit que añadió los resultados) |
| Roca grande en solo 5 de 322 imágenes de M3 | se obtiene sumando la clase por imagen en los `.npz` de M3 |

## 3. Figuras del póster

| Fig. | Fuente | ¿Sale de los datos crudos? |
|---|---|---|
| 1 diagrama del método | `figuras_poster/make_method_diagram.py` → `fig0_method_diagram.png` | No aplica (esquema) |
| 2 barras M1/M2/M3 | `figuras_poster/make_v3_figures.py` → `v3/figA_models.png` | **No: los valores están escritos a mano** en el script (tomados de las tablas) |
| 3 diferencias pareadas | `v3/figB_differences.png` | **No: valores escritos a mano** (tomados de `veredictos_ronda2.csv`) |
| 4 calibración en T | `v3/figD_calibration.png` | Sí, de `raw/e/` |
| 5 IoU por clase y falsos positivos | `v3/figC_perclass.png` | Sí, de `raw/e/` |

La versión de `docs/informe_figuras/fig5_veredictos_forest.png` sí lee los intervalos del CSV de veredictos y comprueba que existan en él.

## 4. Qué se comprobó y qué no

El 19 nov 2026, en CPU y desde los archivos crudos:
- `C_resultados.py` y `F_ronda2_resultados.py` regeneraron **seis tablas de la ronda 1 y tres de la ronda 2 con el mismo contenido** que las versionadas.
- `make_v3_figures.py` regeneró las cuatro figuras del póster **idénticas byte a byte**.
- `make_report_figures.py` generó `docs/informe_figuras/` (curvas de entrenamiento y forest plot).

**Huecos conocidos**
- `A`, `B`, `D` y `E_ronda2` están guardados como `.ipynb` **sin salidas** (solo `C_resultados.ipynb` tiene salidas). No hay logs de consola de la ronda 2 en el repo; su evidencia son los `raw/`.
- El entrenamiento no se reejecutó: requiere Kaggle (≈6 h de GPU en total).
- La barra de DeepLabV3 de la Fig. 2 no tiene un CSV propio.
- Las Figs. 2 y 3 del póster usan valores escritos a mano (coinciden con las tablas, pero el script no los lee de ellas).
- Las cifras de la literatura (valores publicados, Mars-Bench) se tomaron de los artículos y no se recalcularon aquí.
- Las aclaraciones y correcciones del póster impreso están en [`docs/ERRATA_POSTER_V4.md`](docs/ERRATA_POSTER_V4.md).
