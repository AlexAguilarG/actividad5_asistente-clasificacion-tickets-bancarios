# CHANGELOG - Actividad 5

## 0.2.0 - 2026-10-17

- Se incorporaron las métricas agregadas de la ejecución satisfactoria en Google Colab para Regresión Logística y Random Forest.
- Regresión Logística obtuvo F1 macro en prueba de 0.887280, frente a 0.861592 de Random Forest.
- Se incorporó la comparación estadística: McNemar exacto `p = 3.5736e-05` (90 discordancias a favor de Regresión Logística frente a 42 en sentido contrario); Wilcoxon sobre cinco folds `p = 0.0625` y se reporta como exploratorio/no significativo a alpha=0.05.
- Se guardaron `results/model_comparison.csv`, `results/statistical_comparison.json`, `results/RESULTADOS_REALES.md` y dos gráficas de comparación elaboradas a partir de las métricas reportadas.
- Se actualizó la documentación para distinguir la base SQLite de MLflow (`mlflow.db`) del backend legado `mlruns` y evitar abrir la interfaz contra un almacén distinto.
- Se actualizó la conclusión preliminar: Regresión Logística es el candidato preferido para validación adicional; no se afirma aptitud para producción.
- Pendiente: copiar desde Google Drive los reportes detallados por clase, matrices de confusión, predicciones, CSV de Grid Search y cualquier evidencia/captura auténtica del panel de MLflow si se desea distribuirlos con el repositorio.

## 0.1.0 - 2026-10-16

- Se estableció BANKING77 traducido como fuente y se fijó su revisión de Hugging Face.
- Se añadió `fuentes/datos_prep.py` para descarga, validación, limpieza, muestreo y hashes.
- Se añadió `fuentes/train.py` para generar embeddings y comparar Regresión Logística y Random Forest.
- Se definió una búsqueda de hiperparámetros con GridSearchCV y validación cruzada estratificada de cinco folds.
- Se preparó registro de parámetros, métricas, modelos y artefactos con MLflow Tracking.
- Se definieron salidas de métricas estandarizadas, matriz de confusión, comparación estadística y gráficas.

## Política de cambios

Cada modificación de datos, limpieza, encoder, espacio de hiperparámetros o métrica deberá actualizar este archivo e incrementar la versión del experimento. Los CSV con datos personales, la base `mlflow.db` y los artefactos pesados se mantienen fuera de Git por defecto; se versionan los scripts, el notebook y la documentación necesaria para reproducirlos.
