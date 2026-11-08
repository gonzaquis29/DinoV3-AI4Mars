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

## Reformulación del proyecto (02/11/2026)

| # | Decisión | Por qué | IA | Verificó |
|---|---|---|---|---|
| 7 | Tres preguntas: P1 datos (¿la brecha es de apariencia?), P2 modelo (¿dónde adaptar con LoRA?), P3 errores (acuerdo entre expertos y calibración). El teardown es una fase de apoyo, no una pregunta | La tarea pide una pregunta de investigación clara; una sola dirección con tres ángulos | Propuesta y redacción | |
| 8 | Solo AI4Mars; se descarta tamaño del modelo y otros dominios | Profundidad antes que amplitud | — | |
| 9 | Estadística: bootstrap por bloques de 10 imágenes consecutivas, regla de equivalencia (±0.01), corrección de Holm | Las imágenes vecinas (reloj de la nave) se parecen; el bootstrap por imagen sería optimista | Código | |
| 10 | Hipótesis HP1, HP1b, HP2, HP3a, HP3b registradas el 02/11 a las 01:54, antes de correr (`notebooks/estudio_principal/hipotesis_registradas.json`) | Igual que #6 | Redacción | |
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
| 15 | Las hipótesis se registraron de nuevo (`hipotesis_datos_completos.json`) con las mismas reglas, **después** de conocer el piloto y antes de ver resultados completos; esto se declara en el archivo | Honestidad: el piloto ya se conocía | Redacción | |
| 16 | Caché de tokens en disco (memmap) y elección automática de la carpeta con más espacio; si no cabe, se reduce N y queda registrado | `/kaggle/working` solo tiene 20 GB; una primera versión murió con `No space left on device` | Código | |
| 17 | P1: sonda hasta 10 épocas con parada temprana (paciencia 2), las 3 semillas comparten el orden de los lotes | Leer el disco una vez; los IC de las semillas son algo más estrechos que con órdenes independientes | Código | |
| 18 | P2: LoRA con 2 épocas (~1,900 pasos) por corrida, LR fijado con la validación del piloto, 2 semillas, control = sonda de P1 | Lo que cabía en el tiempo; **limitación declarada**, no un diseño ideal | Código | |
| 19 | Baseline entrenado por nosotros: DeepLabV3 ResNet-50 (preentrenado en ImageNet y sin preentrenar), 15,564 imágenes, mismo split y test, 1 semilla, presupuesto de 70 min | La tarea pide posicionar frente a una línea base; las cifras publicadas usan otro split | Código | |
| 20 | El baseline **quedó subentrenado** (3 épocas, ~2,900 pasos). La variante sin preentrenar seguía mejorando | Límite de tiempo; se reporta como limitación, no se oculta | — | |

## Errores del proceso (para la sección de honestidad)

- Piloto con N = 2,000 presentado sin avisar al principio de que difería del póster.
- Dos lanzamientos fallidos en Kaggle por errores del código de lanzamiento (librería subida tarde; caché en un disco de 20 GB).
- Cada fallo se vio solo al terminar el kernel, porque Kaggle no entrega el log antes.

## Ronda 2 (06/11/2026): respuesta a la retroalimentación del profesor

Retroalimentación: (a) H1 y H2 «inconclusas» se ven mal; (b) se sugirió superar un baseline que llega a 0.93 de mIoU (lo leemos como el M3 de SegFormer-B3, 0.93, o de UPerNet Swin-S, 0.94, en Mohammad et al.; **por confirmar con el profesor**).
Los veredictos de la ronda 1 **no se tocan**. Se agrega una ronda 2 con hipótesis nuevas (`notebooks/estudio_principal/hipotesis_ronda2.json`, escrita antes de correr y con lo ya conocido declarado en el archivo).

| # | Decisión | Por qué | IA | Verificó |
|---|---|---|---|---|
| 21 | No cambiar la regla de decisión de HP1/HP2 ni su margen (±0.01) después de ver los resultados | Cambiarlos sería p-hacking. Se agrega una relectura **post hoc**, rotulada como tal: margen 0.02, cotas del efecto («a lo sumo X») y efecto mínimo detectable con 322 imágenes (cuaderno F, parte 1) | Diseño y código | |
| 22 | Diagnóstico: la ronda 1 queda inconclusa por potencia, no por semillas. La σ entre semillas es ~0.001; el IC (±0.01–0.02) lo fija el tamaño del test (322 imágenes). Más semillas no lo arregla | Sale de las tablas P1/P2 | Análisis | |
| 23 | Ronda 2 = diseño 2×2 {decoder congelado, decoder + LoRA en bloques 11–12} × {entropía cruzada, CE + Lovász}, 12 épocas, 3 semillas, mejor época por validación (cuaderno E) | Atribuir la mejora a una pieza. Lovász ataca los falsos positivos de `big_rock`, que explican casi toda la brecha en M3 (probe ~109 k px, decoder ~16 k px). La pérdida es una intervención de Fase B explícita en la tarea | Diseño y código | |
| 24 | Las 4 variantes comparten el tronco congelado y los mismos lotes en cada paso | Coste: un solo pase por el backbone; comparación pareada también en el entrenamiento | Código | |
| 25 | Más épocas: 12 (ronda 1 de LoRA usó 2 y estaba subentrenada). Un ciclo OneCycle, mejor época por validación, checkpoint por época | Más épocas no pueden empeorar lo reportado; Kaggle puede cortar la sesión | Código | |
| 26 | LR sin barrer (decoder 5e-4, LoRA 3e-4); 512 px, no 1024 | Presupuesto de GPU. 1024 px costaría ~4× y no cabe en RAM con la caché uint8. Es una limitación declarada | Decisión del equipo pendiente | |
| 27 | P2b: HP2 repetida con 6 épocas, solo early vs late, 2 semillas (`B_modelo` con `A4M_EPOCHS=6`, resultados `P2L_*`) | HP2 es la hipótesis más débil; el contraste informativo es late−early | Código | |
| 28 | HR3 compara con lo publicado **solo como indicativo** (otro split, otra convención de mIoU, una sola corrida) | Honestidad; el README ya lo dice para la ronda 1 | Redacción | |

Qué esperamos (escrito antes de correr): HR1 apoyada (menos falsos positivos de `big_rock`); HR2 equivalente (LoRA aporta poco sobre un decoder que ya lee varias capas; es el prior honesto y puede volver a ser inconclusa);
HR3: M3 ≥ 0.93 pero **no** superar 0.70 en M1, porque el techo lo pone el ruido de etiquetas; HP2b late > early.

## Resultados de la ronda 2 (07/11/2026)

Kernels (todos COMPLETE, sin errores): `e-run` (8.8 h), `b-t` (8.5 h), `a-t` y `d-t` (lanzados la noche del 06/11, terminaron antes de la mañana del 07/11; `d-t`: 231 min preentrenado, 172 min sin preentrenar), `f` (CPU, solo análisis, `veredictos_ronda2.csv`).
Cambio frente al plan: el entrenamiento quedó en **12,997 imágenes** (no 15,564) por la separación temporal de ≥ 24 h entre entrenamiento y val/T y el chequeo de leakage (`auditoria_datos.json`, sin solapes en ninguna corrida). Evaluación de decisión en **T = 2,000 imágenes**; M1–M3 con 322.
Reglas de `hipotesis_ronda2.json` sin tocar después de ver los datos. `b-long` (P2b) no se lanzó: `b-t` ya cubre early/mid/late con 15 épocas.

| Hipótesis | Veredicto | Lectura honesta |
|---|---|---|
| HR1 Lovász | «Lovász mejor» en IoU de `big_rock` (+0.024 [+0.010, +0.039]) | **El mecanismo previsto no se confirma:** los falsos positivos de `big_rock` *subieron* (T 1.76 M → 2.02 M px; M3 52 k → 91 k). El mIoU global solo sube +0.006 (equivalente). Mejora el IoU de la clase, no por menos FP |
| HR2 LoRA sobre decoder | inconcluso (+0.006 [-0.002, +0.014]) | Cota: a lo sumo +0.014 |
| HR3 vs publicado | «apoyada» (empata en M1, M2, M3) | **Indicativo solo**: «empata» viene de ICs anchos (M3 0.867 [0.739, 0.957]); la cifra puntual de M3 (0.867) está por debajo de 0.93–0.94. Otro split y convención |
| HRA apariencia | inconclusa | clahe −0.010 [-0.018, -0.004] «real»; hist_eq inconcluso; standardize equivalente. Ninguna transformación mejora a la imagen original |
| HRAb blur | inconclusa | blur − original +0.000 [-0.009, +0.009] equivalente en T, pero con cota ±0.009 |
| HRW dónde adaptar | late mejor | late−early +0.033 [+0.023, +0.044] y mid−early +0.030, ambas «reales»; late−mid inconcluso. En M1 todo inconcluso |
| HRB sonda vs DeepLabV3 | inconcluso (−0.005 [-0.019, +0.010]) | DeepLabV3-R50 preentrenado, 1 semilla, 10 épocas útiles, M3 0.884 |

Relectura post hoc de la ronda 1 (`sensibilidad_ronda1.csv`): con margen 0.02 se deciden 9 de 14 comparaciones (con 0.01, 2). Efecto mínimo detectable con 322 imágenes: 0.011 mIoU.
Mejor variante por validación en la ronda 2: `dec_lov` (decoder + Lovász). Contra lo esperado, HR1 no cumple «menos FP» y HRW (late > early) sí cumple el «prior» de HP2b.

Errores del proceso (ronda 2): lanzamientos sin versión quick (fallos solo visibles al terminar); varios entrenamientos de 8–9 h; se usó el CLI de Kaggle de `%APPDATA%` porque el Python de Anaconda del PATH está roto.

## Entrada añadida el 07/11/2026 (decisión de la ronda 2 que no se había registrado)

| # | Decisión | Por qué | IA | Verificó |
|---|---|---|---|---|
| 29 | Crear **T** (2,000 imágenes sacadas del pool de entrenamiento) y decidir todas las hipótesis de la ronda 2 en T; val = 500; train = 12,997 (16,064 − 500 − 2,000 − 567 descartadas por estar a < 24 h de val/T, cifra por diferencia, no medida directamente). val y T son bloques de 100 imágenes consecutivas, al azar, ordenados por reloj real de la nave | La ronda 1 quedó inconclusa por potencia: 322 imágenes dan IC de ±0.011 (M1) a ±0.02 (M2); más semillas no lo arregla (σ ≈ 0.001). T da IC ~2.5× más estrechos | Diseño y código | |

Costos y límites declarados de T: (a) sus etiquetas son las de la multitud, más ruidosas que las de M1–M3, así que su mIoU absoluto no se compara con lo publicado; T es un dispositivo de potencia para comparar modelos entre sí, no un benchmark; (b) comparte distribución con el entrenamiento; (c) la brecha de 24 h es una elección, no está justificada teóricamente; la auditoría halló 6 imágenes de T a ≤ 16 bits de alguna de entrenamiento; (d) se decidió crear T después de conocer los resultados de la ronda 1 (declarado en `hipotesis_ronda2.json`); (e) val y T no son independientes de M1–M3 en la distribución, pero M1–M3 se mantiene intacto y se reporta al lado.
