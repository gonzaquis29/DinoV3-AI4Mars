# Frozen DINOv3 features for Martian terrain segmentation (AI4Mars)

¿Qué tan bien se transfieren las características de un DINOv3 congelado (ViT-B/16, preentrenado en imágenes web) al terreno marciano?
Se estudia sobre **AI4Mars** (Curiosity NavCam; 4 clases: soil, bedrock, sand, big rock) desde tres ángulos:
**apariencia** de la imagen (P1), **dónde adaptar** la red con LoRA (P2) y **dónde y por qué falla** (P3).

**Resultado principal.** Una sonda lineal de 4.6 k parámetros sobre el backbone congelado iguala, dentro del margen estadístico, a un DeepLabV3
entrenado de punta a punta (~8,600 veces más parámetros entrenables). Ni cambiar la apariencia ni adaptar bloques con LoRA mueve el mIoU
más de ~0.01 en los conjuntos de expertos. El límite lo ponen la ambigüedad de las etiquetas y la clase rara `big_rock`.

| Modelo (mIoU test, ronda 1) | Parámetros entrenables | M1 | M2 |
|---|---|---|---|
| Sonda lineal sobre DINOv3 congelado (3 semillas) | 4.6 k | 0.674 | 0.723 |
| DeepLabV3 ResNet-50, preentrenado en ImageNet | 39.6 M | 0.667 | 0.713 |
| DeepLabV3 ResNet-50, sin preentrenamiento | 39.6 M | 0.628 | 0.681 |

## Contenido

```
entregables/   Poster.pdf, Abstract.pdf, Presentacion.pdf, Articulo.pdf (+ fuente LaTeX en articulo_latex/)
notebooks/     Código de los experimentos (ver notebooks/estudio_principal/README.md)
results/       Tablas (.csv), veredictos de las hipótesis, figuras y resultados crudos por corrida
requirements.txt
```

## Reproducir

El entrenamiento corre en Kaggle (2×T4); el análisis corre en CPU a partir de los resultados incluidos:

```bash
pip install -r requirements.txt
A4M_RUN=full python notebooks/estudio_principal/C_resultados.py   # ronda 1
python notebooks/estudio_principal/F_ronda2_resultados.py         # ronda 2
```

## Notas

* Las reglas de decisión de cada hipótesis se escribieron antes de correr (`notebooks/estudio_principal/hipotesis_*.json`).
* La ronda 2 (conjunto interno T, 2,000 imágenes con etiquetas de la comunidad) es **exploratoria**; el criterio principal son los conjuntos M1–M3 de expertos.
* El dataset y los pesos no se incluyen (ver licencias de AI4Mars y DINOv3).
