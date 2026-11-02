# Guía del código: cómo leerlo y cómo explicarlo

Esta guía sirve para entender el estudio principal en 30 minutos y para defenderlo. Todo el estudio vive en `notebooks/estudio_principal/`: **una librería de apoyo y cuatro cuadernos**. Lo demás del repositorio es el estudio anterior (capa por capa) y material de entrega.

## 1. El mapa en una imagen

```
  AI4Mars (imágenes + etiquetas)          DINOv3 ViT-B/16 congelado
              │                                      │
              └────────────── a4m_lib.py ────────────┘
                     datos · tokens en caché · entrenamiento · métricas
                                     │
      ┌───────────────┬──────────────┼───────────────┐
      ▼               ▼              ▼               
 A_datos_y_errores  B_modelo     D_baseline          (GPU, corren en Kaggle)
 teardown, P1,      P2: LoRA     DeepLabV3 de
 datos de P3        por zona     punta a punta
      │               │              │
      └───────────────┴──────┬───────┘
                             ▼
                 results/estudio_principal/raw/   ← un JSON y un .npz por corrida
                             │
                             ▼
                    C_resultados  (CPU, segundos)
          estadística → tablas, figuras y veredictos
                             │
                             ▼
     results/estudio_principal/*.csv, figures/ → entregables/ (artículo, póster), README
```

**La idea de fondo:** los cuadernos de GPU (A, B, D) solo **producen archivos crudos**; el cuaderno C **no entrena nada**, solo lee esos archivos y calcula las tablas. Por eso cualquier número del artículo se puede rastrear hasta un archivo crudo.

## 2. Orden de lectura sugerido

1. `notebooks/estudio_principal/hipotesis_datos_completos.json`: las preguntas, las afirmaciones y las reglas de decisión (5 min).
2. `C_resultados.py`, secciones 1 y 7: cómo se decide «real / equivalente / inconcluso» y cómo salen los veredictos (10 min).
3. `A_datos_y_errores.py`, secciones 2a–2c: las transformaciones de P1 y la sonda (5 min).
4. `B_modelo.py`, secciones 1 y 4: LoRA y una corrida (5 min).
5. `a4m_lib.py`: solo `ReplayNet`, `fit` y `evaluate` (5 min). El resto es contabilidad.

## 3. Archivo por archivo

### `a4m_lib.py` (497 líneas): la tubería compartida
No tiene ideas del estudio; es la contabilidad. Las piezas que conviene conocer:

| Qué | Dónde | Para qué sirve |
|---|---|---|
| `build_splits` | línea 103 | Parte los datos: 500 imágenes de validación (las últimas por reloj de la nave), el resto entrena, y se quitan las que aparecen en test |
| `load_backbone` | línea 196 | Carga DINOv3 ViT-B/16 y lo deja congelado |
| `extract_tokens` | línea 288 | Pasa cada imagen **una sola vez** por el backbone y guarda los tokens (en disco si no caben en RAM) |
| `ReplayNet` | línea 308 | Toma los tokens guardados antes del bloque *k*, ejecuta los bloques *k…12* y aplica la cabeza lineal |
| `per_image_conf`, `miou_np` | líneas 336–348 | Guardan una matriz de confusión **por imagen**; con eso se puede remuestrear después |
| `calib_bins`, `ece` | líneas 361–386 | Estadísticas de confiabilidad y error de calibración |
| `evaluate`, `fit` | líneas 390 y 416 | Evalúa y entrena; `fit` guarda la mejor época según validación y admite parada temprana |
| `save_result` | línea 482 | Escribe `.json` (configuración, mIoU de validación por época) y `.npz` (matrices) |

**Truco central:** como el backbone está congelado, no tiene sentido recalcular sus bloques en cada época. Se calculan una vez y se guardan; entrenar la sonda es entonces casi gratis. En P2, la caché guarda los tokens *justo antes* de los bloques que lleva LoRA, y `ReplayNet` ejecuta solo desde ahí.

### `A_datos_y_errores.py`: teardown, P1 y datos de P3
- **Sección 1:** parámetros, FLOPs, latencia y memoria por bloque.
- **Sección 2a (línea 144):** las cuatro transformaciones: `hist_eq`, `clahe`, `standardize`, `blur`. Son pocas líneas de OpenCV.
- **Sección 2b/2c:** entrena la sonda (4,612 parámetros) con 3 semillas por cada versión de las imágenes. Las tres semillas comparten el orden de los lotes.
- **Datos de P3:** al evaluar cada sonda sobre M1, M2 y M3, `evaluate(..., extra=True)` guarda también, por imagen, las estadísticas de calibración y de acuerdo entre expertos (píxeles etiquetados en M1 y en M3 frente a solo en M1). El análisis de P3 se hace en el cuaderno C; al final de A hay solo una vista previa.

### `B_modelo.py`: P2, LoRA por zona
- **`LoRALinear` (línea 60):** reemplaza `W·x` por `W·x + (B·A·x)·escala`. `B` empieza en cero, así que al inicio el modelo es idéntico al original. Rango 8.
- **`add_lora`:** envuelve cada `nn.Linear` del bloque (q, k, v, salida y las dos del MLP).
- **Sección 4 (una corrida) y 5 (una zona):** entrena con LoRA en dos bloques consecutivos; las zonas son `early` (1–2), `mid` (6–7) y `late` (11–12), con la misma cantidad de parámetros entrenables (221 k).

### `D_baseline.py`: línea base
DeepLabV3-ResNet50 entrenado de punta a punta, con y sin preentrenamiento de ImageNet, con un presupuesto de tiempo (70 min).

### `C_resultados.py`: la estadística y los veredictos (corre en CPU)
Es el cuaderno más importante para defender. Tiene cuatro funciones que lo explican todo:

| Función | Línea | Qué hace |
|---|---|---|
| `block_idx` | 67 | Genera remuestreos formados por **bloques de 10 imágenes consecutivas** |
| `boot` / `point` | 74 / 79 | Métrica promediada sobre semillas, para cada remuestreo |
| `compare` | 83 | Calcula Δ, el IC 95 % por bloques y la etiqueta |
| `holm` | 92 | Corrección por comparaciones múltiples |

La etiqueta de `compare` es la regla del estudio:
- **real:** el IC excluye 0 **y** |Δ| ≥ 0.01
- **equivalente:** el IC cae entero dentro de ±0.01
- **inconcluso:** cualquier otro caso

Está ejecutado y guardado con sus salidas, así que se puede leer sin correr nada.

### `build_kaggle.py`
Arma las carpetas de lanzamiento a Kaggle. No es parte del estudio; se puede ignorar al explicar.

## 4. Seguir un número de principio a fin

Tomemos «la sonda saca 0.674 en M1»:

1. `A_datos_y_errores`, sección 2c, entrena la sonda sobre las imágenes originales con las semillas 0, 1 y 2.
2. Cada corrida escribe `results/estudio_principal/raw/a/P1_original_s0.json` y `.npz` (el `.npz` guarda una matriz de confusión por cada imagen de test).
3. `C_resultados`, sección 2a, suma las matrices por semilla, calcula el mIoU y lo promedia: 0.674.
4. Lo escribe en `results/estudio_principal/P1_table.csv`.
5. El artículo y el póster citan esa fila.

Y los intervalos: se remuestrean los bloques de imágenes (paso 3) 2,000 veces.

## 5. Los conceptos que hay que poder explicar

1. **Por qué una sonda lineal sobre un backbone congelado.** Es la prueba más honesta de «qué tan buenas son las features tal como vienen»; no puede compensar con capacidad propia.
2. **Por qué se guardan tokens.** El backbone no cambia, así que sus salidas se calculan una vez. Es lo que hace viable correr 15,564 imágenes × 3 semillas × 5 transformaciones.
3. **Qué hace LoRA y por qué `B` empieza en cero.** Se agrega una corrección de rango bajo a cada capa lineal; con `B = 0` el punto de partida es el modelo original, así que cualquier cambio viene del entrenamiento.
4. **Por qué bootstrap por bloques.** Las imágenes vecinas del rover (por reloj de la nave) se parecen; remuestrear imagen por imagen daría intervalos demasiado optimistas.
5. **Qué significa «equivalente» frente a «inconcluso».** Equivalente: sabemos que la diferencia es menor que 0.01. Inconcluso: no sabemos, el intervalo es demasiado ancho. No es lo mismo que «cero».
6. **Qué mide el ECE y por qué se mira aparte en `big_rock`.** El ECE compara confianza con acierto por intervalos de confianza. Un valor global bueno (0.029) puede esconder que en una clase rara el modelo está muy seguro y se equivoca (+0.674).

## 6. Preguntas probables y respuestas cortas

- **¿Por qué solo 2 épocas en LoRA?** Por el tiempo de GPU disponible. Está declarado como limitación; el control (la sonda) usa la misma receta, y en validación las zonas sí se separan, cosa que no explicamos.
- **¿Por qué el baseline llega a 0.667 y no a más?** Es un presupuesto igualado de 3 épocas; el de ImageNet ya se había estancado, el que no está preentrenado seguía subiendo.
- **¿Por qué ±0.01?** Es el umbral que decidimos antes de correr; por debajo no nos parece una diferencia que cambie una conclusión práctica.
- **¿Se eligió algo con el test?** No: capa, época y tasa de aprendizaje se eligen en validación.
- **¿Qué hace el piloto de 2,000 imágenes?** Es una réplica independiente; llevó a las mismas conclusiones. Las reglas se volvieron a registrar después de conocerlo, y está declarado.
- **¿Qué falta?** Campo receptivo y visualización de features, un barrido de hiperparámetros y corridas más largas de LoRA y del baseline.

## 7. Por qué el repositorio parece desordenado

Hay dos generaciones de código y varias capas de andamiaje:

- **Estudio anterior (capa por capa):** `notebooks/ai4mars_dinov3_final.py` es un cuaderno único y largo (etapas A–E); sus resultados están en `results/final_run/`. El estudio principal **no depende de él**; solo lo cita como contexto.
- **Estudio principal:** `notebooks/estudio_principal/` con la librería y los cuatro cuadernos. Es lo que hay que leer.
- **Modos de ejecución:** variables de entorno como `A4M_FULL`, `A4M_QUICK`, `A4M_SMOKE` seleccionan entre corrida completa (15,564 imágenes), prueba rápida en Kaggle y autoprueba con datos falsos. El piloto de 2,000 imágenes corre cuando no se define `A4M_FULL`. Si solo se quiere entender, se pueden ignorar.
- **`kaggle_run/`:** carpetas de lanzamiento a Kaggle generadas por `build_kaggle.py`.

Si hay que simplificar la explicación oral: «hay una librería de apoyo, tres cuadernos que producen archivos crudos en GPU y uno que los analiza en CPU».
