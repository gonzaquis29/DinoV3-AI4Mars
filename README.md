# DinoV3-AI4Mars

Segmentación de terreno marciano con **DINOv3** sobre el dataset **AI4Mars**.

## Idea
Usar DINOv3 como backbone (congelado o con ajuste ligero) para segmentar el terreno que ven los rovers de Marte (suelo, roca madre, arena, roca grande) y evaluar cuánto aportan sus features densos frente a una línea base supervisada clásica.

## Preguntas
- ¿Qué tan bien segmenta un backbone DINOv3 congelado con una cabeza lineal, frente a una U-Net/DeepLab entrenada desde cero?
- ¿Cuánto mejora con pocas etiquetas (few-shot: 1%, 10%, 100%)?
- ¿Qué tan bien generaliza entre misiones/rovers (por ejemplo, entrenar en Curiosity y probar en Opportunity/Spirit)?
- ¿Qué tamaño de DINOv3 (ViT-S/B/L) es suficiente?

## Plan (borrador)
1. Descargar AI4Mars y explorar clases, desbalance y ruido de etiquetas.
2. Línea base: U-Net o DeepLab.
3. DINOv3 congelado + cabeza lineal / ligera sobre features densos.
4. Experimentos de pocas etiquetas y de generalización entre rovers.
5. Métricas: mIoU por clase, accuracy por píxel; análisis cualitativo de errores.

## Estructura
- `data/`: dataset (no se versiona)
- `notebooks/`: exploración y análisis
- `src/`: código de entrenamiento y evaluación
- `results/`: métricas, figuras, checkpoints ligeros

## Por definir
- Curso/entrega y fechas
- GPU disponible
- Versión exacta de AI4Mars y de los pesos de DINOv3
