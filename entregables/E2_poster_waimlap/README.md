# E2: póster y resumen extendido para WAIMLAp

| Carpeta | Versión | Qué presenta |
|---|---|---|
| [`version_octubre/`](version_octubre/) | Versión de octubre (registro) | El estudio capa por capa: sonda lineal por capa, decoder, ajuste fino parcial y controles (DINOv2, ViT aleatorio) |
| [`version_final/`](version_final/) | Versión final | Los resultados combinados: el estudio por capas (sonda por capa, decoder, ajuste fino, controles) y el estudio principal (P1 apariencia, P2 LoRA por profundidad, P3 errores y calibración, baseline DeepLabV3) |

Cada carpeta tiene el resumen extendido (2 páginas, plantilla de WAIMLAp) en `.docx` y `.pdf`, y el póster A0 en `.pptx` y `.pdf`.
`version_final/` incluye además `Presentacion_Avance` (`.pptx` y `.pdf`), una presentación de 16 láminas más un anexo, que cubre el estudio por capas y el estudio principal.
Los números de la versión final salen de `results/estudio_principal/`.

**Póster v4 (WAIMLAp, 19/11/2026).** `version_final/Poster_FrozenDINOv3MartianTerrain_v4.pdf` y `.pptx` es el póster impreso: añade la ronda 2 (conjunto T, `results/estudio_principal/`). El resumen extendido de `version_final/` es anterior a la ronda 2 y no la incluye.

**Título (19/11/2026).** `Poster_..._v4_enviado.pdf` es el PDF que se mandó a los organizadores (título "How Do Web-Pretrained DINOv3 Features Extend to Martian Terrain?..."). `Poster_..._v4.pdf` y `.pptx` llevan el título actualizado ("Frozen DINOv3 Features for Martian Terrain Segmentation: A Study on AI4Mars"); es la única diferencia de contenido. Ver [errata, sección 0](../../docs/ERRATA_POSTER_V4.md).

**Versión final (19/11/2026).** Los archivos `Poster_..._v4.pdf` y `.pptx` de esta carpeta son la versión final presentada.

**Correcciones (19/11/2026).** El `.pdf`/`.pptx` final de `version_final/` lleva el título "Frozen DINOv3 Features for Martian Terrain Segmentation: A Study on AI4Mars" y correcciones de texto y pies de figura (citas, H2 exploratoria, origen del 0.674, tamaño de los modelos publicados, etc.). Detalle en [errata, secciones 0, 0b y 0c](../../docs/ERRATA_POSTER_V4.md). El PDF enviado se conserva como `_enviado.pdf`.
