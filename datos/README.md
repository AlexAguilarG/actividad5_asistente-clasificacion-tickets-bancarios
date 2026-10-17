# Datos

La fuente se descarga de `DeepPavlov/banking77-translated`, configuración `es`, revisión `48246f6c99b06096d9dbd7fcbaf273c6d82df2f0`.

- `datos_ini/`: contiene la exportación original de las particiones del dataset cuando se ejecuta `python fuentes/datos_prep.py`.
- `datos_limp/`: contiene las muestras preparadas `train_limp.csv` y `test_limp.csv` que usa el entrenamiento.
- `diccionario_intenciones.csv`: IDs canónicos de las 77 intenciones.

Los CSV descargados no se guardan en Git por defecto. El proceso de obtención y limpieza está en `fuentes/datos_prep.py`; el reporte de calidad y hashes se guardan en `results/data_quality_report.json`.
