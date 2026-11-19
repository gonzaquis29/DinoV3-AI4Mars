# ¿Sirve una IA entrenada con fotos de internet para entender el suelo de Marte?

*DINOv3 congelado en AI4Mars. Versión corta, 08/11/2026. Los números salen de los resultados incluidos en el repositorio.*

> Aclaraciones y correcciones de este texto y del póster impreso: [`ERRATA_POSTER_V4.md`](ERRATA_POSTER_V4.md).

---

## 1. Contexto

Un rover como Curiosity necesita reconocer qué tiene delante (suelo firme, roca madre, arena o rocas grandes) para decidir por dónde avanzar sin atascarse ni chocar. Entrenar una IA para esa tarea requiere miles de fotos marcadas píxel por píxel, y recolectar esas marcas es un proceso costoso en tiempo y en personas. El proyecto AI4Mars lo resolvió con voluntarios por internet y reservó un conjunto pequeño de fotos marcadas por expertos para usarlo como examen.

En paralelo han surgido modelos de fundación de visión como DINOv3, un Vision Transformer (ViT-B/16) entrenado de forma autosupervisada con 1.689 millones de imágenes de la Tierra, sin etiquetas. Sus representaciones han resultado útiles en muchas tareas, pero no se sabe si se extrapolan a dominios muy distintos de aquel en que se entrenaron, como la superficie de Marte, cuyas fotos son en blanco y negro, no contienen objetos humanos y muestran terrenos muy parecidos entre sí. Si lo hicieran, se necesitaría mucho menos cómputo y menos etiquetado, algo valioso en una misión con energía y comunicación limitadas. El problema que motiva este trabajo es, entonces, saber cuánto de ese conocimiento sirve en Marte sin reentrenar el modelo, y cuánto ajuste adicional hace falta.

## 2. Antecedentes

Revisamos once trabajos a texto completo. Los modelos entrenados específicamente para AI4Mars, con redes de 47 a 81 millones de parámetros, obtienen entre 0.67 y 0.70 en el examen de expertos, usando una nota llamada mIoU que va de 0 a 1 y mide cuánto se solapa lo predicho con lo marcado; su punto más débil son las rocas grandes (Mohammad y colaboradores, 2025). Goh y colaboradores (2022) mostraron que partir de un modelo preentrenado mejora mucho el resultado cuando hay pocas fotos marcadas. Sin embargo, midieron el porcentaje de píxeles acertados y no el mIoU, así que sus cifras no son comparables con las nuestras.

Sobre modelos de fundación aplicados a Marte la literatura es escasa. MANTLE (2026) usó DINOv2 sin modificarlo, pero en otro conjunto de fotos y para detectar rocas, una tarea más simple. El benchmark Mars-Bench (2025) probó modelos congelados, pero descartó AI4Mars porque las etiquetas de voluntarios le parecieron poco confiables. Y MarsUnet (2026) probó adaptadores sobre una red convolucional antigua, con tres clases en lugar de cuatro y sin el examen de expertos.

## 3. Gap

La literatura no contiene una evaluación controlada de un modelo de fundación congelado, como DINOv3, sobre AI4Mars con el examen de expertos. Tampoco compara, con el mismo modelo y los mismos datos, cuánto conviene adaptarlo, ni distingue las mejoras reales del ruido: los resultados publicados son un solo número por modelo, medido sobre apenas 322 fotos, donde diferencias de 0.01 a 0.03 caen dentro del error de medición.

## 4. Problema

> No se sabe cuánto de lo aprendido con fotos de la Tierra transfiere un modelo de fundación congelado al terreno marciano, qué parte conviene adaptar para mejorarlo, ni con qué certeza se pueden distinguir esas mejoras.

De aquí salen tres preguntas. La primera, P1, es si la dificultad está en cómo se ven las fotos, por ejemplo su contraste o su textura. La segunda, P2, es en qué parte del modelo conviene concentrar el ajuste. La tercera, P3, es cuánto del error se debe a que las etiquetas son ambiguas y si el modelo sabe cuándo se equivoca.

## 5. Objetivos

**Objetivo general.** Determinar hasta qué punto DINOv3 congelado sirve para segmentar el terreno de Marte y qué adaptaciones mejoran de verdad el resultado, indicando qué diferencias el examen puede detectar y cuáles no.

**Objetivos específicos.**
1. Medir el rendimiento de DINOv3 congelado con cabezas simples, sin modificar el modelo.
2. Evaluar si alterar el contraste o la textura de las fotos mejora el resultado (P1).
3. Comparar el ajuste de capas tempranas, medias y tardías con el mismo esfuerzo (P2).
4. Cuantificar el techo que imponen las etiquetas ambiguas y la confianza del modelo en las rocas grandes (P3).

## 6. Hipótesis

El estudio plantea tres hipótesis, una por cada pregunta. Cada una se escribió, con su regla de decisión, antes de correr los experimentos. La regla dice que una diferencia es "real" si supera 0.01 de nota y su intervalo de confianza no incluye el cero, que "no hay diferencia" si todo el intervalo cae dentro de ±0.01, y que el resultado es "inconcluso" en cualquier otro caso.

| | Esperábamos que… |
|---|---|
| **H1** (P1) | cambiar el contraste o la textura de las fotos no mejore el resultado, es decir, que la dificultad no esté en la apariencia. |
| **H2** (P2) | ajustar las capas profundas ayude más que las iniciales. |
| **H3** (P3) | buena parte del error venga de etiquetas ambiguas, y que el modelo esté demasiado seguro cuando se equivoca con rocas grandes. |

Después de la primera ronda, a raíz de la retroalimentación del profesor (varias conclusiones habían quedado "inconclusas" y se sugería compararse con lo publicado), añadimos pruebas que no son hipótesis nuevas ni objetivos, sino comprobaciones puntuales con regla escrita de antemano. Dos son complementarias: C1, si una función de pérdida pensada para clases raras mejora las rocas grandes con menos falsas alarmas, y C2, si ajustar la IA sobre una versión ya más completa aporta poco. Las otras dos son comparaciones de referencia (baselines): C3, si la versión más simple iguala a una red clásica entrenada completa, y C4, si nuestro mejor modelo alcanza las cifras publicadas. Esta segunda ronda se diseñó conociendo ya los resultados de la primera, lo cual queda declarado en el archivo de registro.

## 7. Metodología

La idea es tomar DINOv3 sin modificarlo y agregarle una "cabeza" pequeña que traduzca lo que ve en las cuatro clases, para medir qué tan bien lo hace con distintas cabezas y ajustes. Usamos fotos de la cámara de navegación de Curiosity (AI4Mars), en blanco y negro y a 512×512 píxeles, porque es el único conjunto de Marte con un examen de expertos y comparaciones publicadas.

Las fotos se reparten en entrenamiento, validación (para elegir ajustes) y dos exámenes: el de expertos (322 fotos, en tres versiones M1, M2 y M3 según cuántos expertos coincidan) y un examen interno, T, de 2.000 fotos. Creamos T porque con 322 fotos el intervalo de confianza de una diferencia entre dos modelos mide entre ±0.011 y ±0.02, y las diferencias que nos interesan (de 0.01 a 0.03 de mIoU) quedan dentro de ese margen. T se saca del conjunto de entrenamiento, así que ese tamaño es un compromiso: con 2.000 fotos los intervalos se estrechan unas 2,5 veces y todavía quedan 12.997 fotos para entrenar. No hicimos un cálculo formal de potencia; el tamaño se eligió así, tras ver que la primera ronda no alcanzaba para decidir. Sus etiquetas son de voluntarios, así que T sirve para comparar nuestros modelos entre sí, no para compararnos con la literatura. Para evitar trampas, las fotos se ordenan por la hora real de la nave y se descarta del entrenamiento toda foto a menos de 24 horas de una de examen, ya que fotos casi simultáneas son casi idénticas e inflarían la nota. Un programa verifica esto antes de cada corrida y la detiene si encuentra problemas; en todas las corridas encontró cero fotos repetidas.

Probamos tres cabezas de complejidad creciente: una sonda lineal (4.612 piezas ajustables, lo más simple posible), un decodificador (3,5 millones) y LoRA, que además retoca ligeramente algunas capas internas de DINOv3 (221 mil piezas). La sonda actúa como piso, y las otras muestran cuánto aporta dar más ayuda. Todos se entrenan con las mismas reglas, se repiten con 2 o 3 semillas aleatorias, y la mejor época de entrenamiento se elige siempre según validación, nunca según el examen, para no hacer trampa. Cada pregunta cambia una sola cosa: para P1 se entrena con la foto original y con cuatro versiones alteradas (ecualización, CLAHE, estandarización y desenfoque); para P2 se aplica LoRA en capas tempranas, medias o tardías con la misma cantidad de piezas, de modo que cambie el dónde y no el cuánto; para P3 se analiza dónde coinciden los expertos y se prueba la pérdida especial para rocas grandes; y como referencia se entrena una red clásica (DeepLabV3), con y sin preentrenamiento, y se contrastan los resultados con los publicados.

Medimos con mIoU, que es más exigente que el porcentaje de aciertos porque no deja que las clases raras queden ocultas. Para cada diferencia calculamos un intervalo de confianza remuestreando grupos de diez fotos consecutivas (las vecinas se parecen) y aplicamos la regla de la sección 6, lo que evita celebrar diferencias que son ruido. Todo se corrió en Kaggle con dos tarjetas T4, unas 30 horas de GPU en la segunda ronda; los cuadernos y los resultados por corrida están en `notebooks/estudio_principal/` y `results/estudio_principal/`. Con los mismos datos y semillas deberían obtenerse las mismas conclusiones, pero no los mismos decimales, porque no probamos una repetición completa. El estudio tuvo dos rondas: en la primera varias hipótesis quedaron inconclusas por el tamaño del examen, y la segunda añadió T para poder decidir.

## 8. Resultados

El resultado central es que casi todo ya viene incluido en DINOv3. La sonda más simple, con apenas 4.612 piezas ajustables, queda dentro del rango de modelos miles de veces más grandes.

| Modelo | Piezas ajustables | Nota en M1 | Nota en M2 |
|---|---|---|---|
| Sonda lineal | 4.612 | 0.674 | 0.723 |
| Decodificador | 3,5 millones | 0.682 | 0.732 |
| Red clásica DeepLabV3 entrenada completa | 39,6 millones | 0.675 | 0.727 |
| *Mejores publicados* | *47 a 81 millones* | *0.69–0.70* | *0.74* |

Sonda, decodificador y ajustes parciales quedan a menos de 0.012 entre sí: el rendimiento se estanca, y agregar capacidad ya no lo mueve. El resultado de las tres hipótesis, las pruebas complementarias (C1, C2) y las comparaciones de referencia (C3, C4) fue el siguiente.

| Hipótesis, prueba o referencia | Qué pasó |
|---|---|
| **H1** Cambiar las fotos | Ninguna alteración mejora, y el CLAHE empeora levemente (−0.010). El veredicto formal es inconcluso, pero la conclusión práctica es clara. |
| **H2** Dónde ajustar | Ajustar capas medias o tardías rinde unos +0.03 más que las iniciales en T. En el examen de expertos no se logra confirmar y hay una salvedad importante (sección 9). |
| **H3** Etiquetas y confianza | Confirmada. Donde los expertos coinciden la nota es 0.86; donde solo uno marcó, 0.55. En rocas grandes el modelo está muy seguro y falla en cerca de la mitad de los píxeles. |
| **C1** Pérdida para rocas grandes | Mejora el solape de esa clase (+0.024), pero no por menos falsas alarmas: aumentaron. Detecta más rocas, pero también inventa más. |
| **C2** Más ajuste sobre el decodificador | Aporta a lo sumo +0.014; no se puede decidir. |
| **C3** Referencia: simple vs. red clásica | Empate estadístico (−0.005), sin poder decidir; la red clásica se entrenó una sola vez. |
| **C4** Referencia: contra lo publicado | Nuestras notas son algo menores en los tres exámenes (en M3, 0.867 frente a 0.93). Por la regla "empatamos" porque los intervalos son muy anchos, pero no superamos a nadie. |

## 9. Discusión y conclusión

Los resultados sugieren que un modelo de fundación ya entiende bastante de Marte, y que retocarlo mejora poco. Sospechamos que el límite está en las etiquetas, no en el modelo: la diferencia entre píxeles con acuerdo de expertos y sin él (0.86 contra 0.55) es mayor que cualquier ajuste que probamos. El punto débil son las rocas grandes, que son raras. Para un rover, inventar una roca (frenar sin necesidad) y no verla (chocar) no cuestan lo mismo, y el mIoU no distingue entre ambos errores.

Hay varias salvedades que conviene tener presentes. La más seria afecta a H2: a la zona temprana le dimos una velocidad de aprendizaje menor que a las otras dos, y su curva se estanca pronto, de modo que parte de la ventaja de "ajustar profundo" podría deberse a esa configuración y no a la profundidad; falta una corrida de control. Además, esa ventaja aparece en T, cuyas etiquetas comparten el estilo del entrenamiento, y no en el examen de expertos, así que el modelo podría estar aprendiendo el estilo de los voluntarios. Entre rondas, la nota en M3 bajó de 0.93 a 0.87 sin una explicación verificada; M3 tiene rocas grandes en solo cinco fotos, lo que la vuelve muy inestable. Finalmente, hicimos pocas repeticiones (2 o 3, y una sola para la red clásica), usamos una cámara, un rover y una IA, y trabajamos a 512 píxeles en lugar de los 1024 de otros estudios.

En conclusión, un modelo de fundación congelado, con muy pocas piezas ajustables, llega al nivel de modelos mucho más grandes en terreno marciano. El techo parece venir de las etiquetas más que del modelo; cambiar contraste o textura no ayuda; ajustar capas profundas ayuda algo, con la salvedad indicada; y las rocas grandes siguen siendo el punto débil, con un modelo que no sabe que se equivoca. No superamos lo publicado: nuestro aporte es de eficiencia (casi igual con muchísimo menos) y de rigor (reglas escritas de antemano, intervalos de confianza y chequeo de trampas), no de mayor exactitud. Los próximos pasos son repetir la zona temprana con la misma velocidad de aprendizaje, hacer más repeticiones, conseguir un examen de expertos más grande y probar 1024 píxeles, otros tamaños de DINOv3 y otros rovers.

*Uso responsable: son resultados de laboratorio con datos públicos de NASA/JPL y no bastan por sí solos para decidir la seguridad de un rover real. El código se escribió con ayuda de IA (Claude); las decisiones están en `DESIGN_LOG.md`.*

*Referencias principales: Siméoni et al. 2025 (DINOv3); Swan et al. 2021 (AI4Mars); Mohammad, Li y Gao 2025; Goh et al. 2022; Durai y Doran 2026 (MANTLE); Purohit et al. 2025 (Mars-Bench); Dai et al. 2026 (MarsUnet); Hu et al. 2022 (LoRA).*
