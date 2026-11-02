# E3: entrega de trabajo y artículo de taller

## Artículo (30/11/2026)
[`articulo/`](articulo/): fuente LaTeX (`main.tex`), clase IEEE (`IEEEtran.cls`) y PDF compilado (`main.pdf`, 5 páginas). Es el borrador actual; el bloque de autores sigue como marcador.
Para compilarlo: `cd entregables/E3_trabajo_y_articulo/articulo && pdflatex main.tex` (dos veces). Las figuras se leen de `results/estudio_principal/figures/`.

## Entrega de trabajo (16/11/2026)
La entrega de trabajo es este repositorio:

| Parte | Dónde |
|---|---|
| Cuadernos reproducibles y semillas fijas | `notebooks/estudio_principal/` (estudio final) y `notebooks/ai4mars_dinov3_final.ipynb` (estudio anterior) |
| Resultados, tablas, figuras y veredictos | `results/estudio_principal/` (y `results/estudio_principal_piloto/`, `results/final_run/`) |
| Logs de las corridas en GPU y archivos crudos | `results/estudio_principal/logs/` y `results/estudio_principal/raw/` |
| Cuaderno de análisis ejecutado | `notebooks/estudio_principal/C_resultados.ipynb` y `docs/Resultados_ejecutado.html` |
| Cuaderno de decisiones (con uso de IA) | `DESIGN_LOG.md` |
| Instrucciones, entorno y semillas | `README.md`, `requirements.txt` |
| Guía para leer y explicar el código | `docs/GUIA_DEL_CODIGO.md` |
