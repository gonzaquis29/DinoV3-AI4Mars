# Cuaderno de decisiones (design log)

Proyecto: ¿cómo se extienden a Marte (AI4Mars) las features de DINOv3 preentrenado con imágenes web?
Cada entrada: decisión, por qué, y dónde se usó asistencia de IA. Las celdas «Verificó» las completa el equipo.

Política de IA de la tarea: el código de los notebooks y de los scripts de lanzamiento lo escribió **Claude (Anthropic)**, a petición del equipo.
Delegar la ejecución es válido; la comprensión no. Cada integrante debe poder explicar cualquier decisión de esta tabla en la defensa oral.

## Corrida final (22/10/2026)

| # | Decisión | Por qué | IA | Verificó |
|---|---|---|---|---|
| 1 | Protocolo unificado con el split del experimento de otro integrante del equipo: 500 imágenes de validación temporal, 15,564 para entrenar; M1, M2 y M3 por separado | Comparabilidad entre integrantes; M3 tiene pocas imágenes con `big_rock`, no se mezcla | Código | |
| 2 | Caché de imágenes uint8 y aumento en GPU | El cargador original iba a ~20 img/s; con caché ≈ 370 img/s | Código | |
| 3 | Decoder reentrenado con 3 semillas y 12 épocas, no se reutiliza el checkpoint de 20 épocas | Mismo pipeline para todas las comparaciones | Código | |
| 4 | Selección de capa y de época solo por validación, nunca por test | Evitar optimismo | Código | |
| 5 | «Diferencia real» = IC 95 % bootstrap pareado excluye 0 **y** \|Δ\| ≥ 0.01 mIoU | Evitar celebrar diferencias triviales | Diseño y código | |
| 6 | Hipótesis H1–H6 escritas antes de correr | Se evalúa el modelo mental, no la ejecución | Redacción | |

## Reformulación v2 (02/11/2026)

| # | Decisión | Por qué | IA | Verificó |
|---|---|---|---|---|
| 7 | Tres preguntas: P1 datos (¿la brecha es de apariencia?), P2 modelo (¿dónde adaptar con LoRA?), P3 errores (acuerdo entre expertos y calibración). El teardown es preliminar, no una pregunta | La tarea pide una pregunta de investigación clara; una sola dirección con tres ángulos | Propuesta y redacción | |
| 8 | Solo AI4Mars; se descarta tamaño del modelo y otros dominios | Profundidad antes que amplitud | — | |
| 9 | Estadística: bootstrap por bloques de 10 imágenes consecutivas, regla de equivalencia (±0.01), corrección de Holm | Las imágenes vecinas (reloj de la nave) se parecen; el bootstrap por imagen sería optimista | Código | |
| 10 | Hipótesis HP1, HP1b, HP2, HP3a, HP3b registradas el 02/11 a las 01:54, antes de correr (`notebooks/v2/hypotheses_v2.json`) | Igual que #6 | Redacción | |
| 11 | LoRA rango 8 sobre todas las capas lineales de 2 bloques consecutivos (1–2, 6–7, 11–12), mismo presupuesto en las tres zonas | Aislar *dónde* adaptar, no *cuánto* | Código | |

## Piloto con N = 2,000 (02/11/2026, mañana)

El piloto entrenó con **2,000 imágenes** (el pool de la etapa D), no con las 15,564. Se decidió así para terminar de madrugada.
No se avisó al equipo de que eso no coincidía con el entrenamiento completo del póster. Se corrigió el mismo día (ver siguiente sección).

| # | Decisión | Resultado |
|---|---|---|
| 12 | Primera corrida de A falló por RAM del host (DataLoader con procesos hijos, 4.ª transformación). Se pasó a `num_workers=0`, se liberó la caché entre transformaciones y se reanudó con la salida previa como dataset | A completo: 5 transformaciones × 3 semillas |
| 13 | Resultados del piloto: HP1, HP1b, HP2 **inconclusas**; HP3a y HP3b **apoyadas** | Se conservan como apéndice, claramente marcados como N = 2,000 |

## Versión final con las 15,564 imágenes (02/11/2026)

| # | Decisión | Por qué | IA | Verificó |
|---|---|---|---|---|
| 14 | Rehacer P1, P2 y P3 con todas las imágenes de entrenamiento | El entregable final y el póster usan el conjunto completo | Código | |
| 15 | Las hipótesis se registraron de nuevo (`hypotheses_v2_full.json`) con las mismas reglas, **después** de conocer el piloto y antes de ver resultados completos; esto se declara en el archivo | Honestidad: el piloto ya se conocía | Redacción | |
| 16 | Caché de tokens en disco (memmap) y elección automática de la carpeta con más espacio; si no cabe, se reduce N y queda registrado | `/kaggle/working` solo tiene 20 GB; una primera versión murió con `No space left on device` | Código | |
| 17 | P1: sonda hasta 10 épocas con parada temprana (paciencia 2), las 3 semillas comparten el orden de los lotes | Leer el disco una vez; los IC de las semillas son algo más estrechos que con órdenes independientes | Código | |
| 18 | P2: LoRA con 2 épocas (~1,900 pasos) por corrida, LR fijado con la validación del piloto, 2 semillas, control = sonda de P1 | Lo que cabía en el tiempo; **limitación declarada**, no un diseño ideal | Código | |
| 19 | Baseline entrenado por nosotros: DeepLabV3 ResNet-50 (preentrenado en ImageNet y sin preentrenar), 15,564 imágenes, mismo split y test, 1 semilla, presupuesto de 70 min | La tarea pide posicionar frente a una línea base; las cifras publicadas usan otro split | Código | |
| 20 | El baseline **quedó subentrenado** (3 épocas, ~2,900 pasos). La variante sin preentrenar seguía mejorando | Límite de tiempo; se reporta como limitación, no se oculta | — | |

## Errores del proceso (para la sección de honestidad)

- Piloto con N = 2,000 presentado sin avisar al principio de que difería del póster.
- Dos lanzamientos fallidos en Kaggle por errores del código de lanzamiento (librería subida tarde; caché en un disco de 20 GB).
- Cada fallo se vio solo al terminar el kernel, porque Kaggle no entrega el log antes.
