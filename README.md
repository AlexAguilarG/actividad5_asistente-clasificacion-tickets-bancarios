# Actividad 5. Preparación, entrenamiento, ajuste y trazabilidad con MLflow

## Descripción del proyecto

Este proyecto continúa la Actividad 4: un prototipo de procesamiento de lenguaje natural (PLN) para clasificar mensajes de banca digital por intención. Se utiliza la variante española de BANKING77 publicada como `DeepPavlov/banking77-translated` y se comparan dos algoritmos sobre la misma representación semántica: **Regresión Logística** y **Random Forest**. Los textos se transforman en embeddings con `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; el codificador permanece congelado para que la comparación mida la diferencia entre los clasificadores.

La actividad incorpora limpieza trazable, muestra reproducible, validación cruzada estratificada de cinco folds, búsqueda de hiperparámetros (`GridSearchCV`), métricas estandarizadas, prueba estadística pareada y registro de parámetros, métricas, modelos y artefactos en MLflow.

**Alcance:** prototipo académico; no se conecta a sistemas del banco ni ejecuta operaciones financieras. BANKING77 traducido no es un histórico de tickets de un banco mexicano. La ejecución satisfactoria del 17 de octubre de 2026 favoreció a Regresión Logística frente a Random Forest en las seis métricas agregadas compartidas. Las métricas de la Actividad 4 siguen siendo únicamente referencia histórica y no sustituyen los resultados de esta actividad.

## Estructura

```text
Actividad5/
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── .gitignore
├── datos/
│   ├── datos_ini/                 # CSV originales descargados; ignorados por Git
│   ├── datos_limp/                # CSV limpios/muestreados; ignorados por Git
│   └── diccionario_intenciones.csv
├── fuentes/
│   ├── entrena.ipynb              # notebook para ejecutar los scripts
│   ├── datos_prep.py              # descarga, limpieza, muestreo y perfil de datos
│   └── train.py                   # Grid Search, evaluación y registro MLflow
├── results/                       # tablas, gráficas, matrices, reportes y comparación estadística
├── modelos/                       # clasificadores serializados, ignorados por Git
├── mlflow.db                      # MLflow Tracking SQLite, ignorado por Git
├── evidencias/
│   └── mlflow/                    # capturas manuales reales del panel MLflow
└── docs/
    └── REPORTE_TECNICO_ACTIVIDAD5.pdf
```

## Dataset y decisiones de preparación

- Fuente: `DeepPavlov/banking77-translated`, configuración `es`, revisión fijada `48246f6c99b06096d9dbd7fcbaf273c6d82df2f0`.
- El texto del experimento proviene de `text_gemma`; la etiqueta objetivo es `label` (IDs 0-76).
- La partición `train` y `test` de la fuente se mantienen separadas; no se mezclan antes del muestreo.
- Limpieza: trim de espacios exteriores, descarte de texto vacío/nulo, conversión de etiqueta a entero y descarte de etiquetas fuera de 0-76.
- Se registran los duplicados exactos `text`+`label`, pero no se borran automáticamente: esto evita una modificación silenciosa del benchmark. El reporte de calidad deja constancia para decidir si deben eliminarse.
- Muestreo reproducible con semilla 42: máximo 80 ejemplos por clase en train y 25 por clase en test, conservando las particiones originales. Los tamaños efectivos y hashes se escriben en `results/data_quality_report.json`.
- Los CSV descargados no se incluyen en el repositorio de Git por defecto. Los scripts constituyen el mecanismo reproducible de obtención y evitan redistribuir datos sin confirmar las condiciones de la variante traducida.

## Entorno de ejecución

La ejecución reportada se realizó en Google Colab con Python 3.13 y GPU CUDA para generar embeddings. Para reproducir, utiliza el mismo `requirements.txt` y conserva la información de versiones registrada en MLflow. Google Colab con GPU acelera la generación de embeddings; CPU también es posible, con mayor duración. Instala las dependencias desde el directorio raíz:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Pasos de ejecución

Desde el directorio raíz `Actividad5/`:

```bash
python fuentes/datos_prep.py
python fuentes/train.py
```

El primer script descarga el dataset fijando la revisión, valida columnas y particiones, prepara muestras por clase y genera el perfil de calidad. El segundo descarga el encoder de Sentence Transformers, crea embeddings una sola vez, ajusta ambos algoritmos, evalúa el conjunto de prueba y registra cada experimento en MLflow.

### Panel de MLflow

El script de entrenamiento configura MLflow con una base de datos SQLite en `mlflow.db` (no con el backend antiguo de carpeta `mlruns`). Para abrir el panel local desde la raíz del repositorio:

```bash
mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow.db" --host 127.0.0.1 --port 5000
```

Después abre `http://127.0.0.1:5000`. En Google Colab, ejecuta la celda de interfaz incluida en `fuentes/entrena.ipynb`; esta debe utilizar exactamente la URI `sqlite:////.../Actividad5/mlflow.db` que mostró el entrenamiento. No abras un panel apuntando a `mlruns`, porque sería un almacén distinto.

### Notebook de Colab

Abre `fuentes/entrena.ipynb` en Colab con el repositorio disponible en `/content/Actividad5` (puedes subir/descomprimir el ZIP o clonar el repositorio). Ejecuta las celdas de arriba abajo. El notebook instala dependencias, ejecuta la preparación y el entrenamiento, y presenta las tablas de resultados.

## Algoritmos e hiperparámetros

**Regresión Logística:** `C ∈ {0.1, 1, 10}`; `class_weight ∈ {None, balanced}`; `max_iter=2500`; solver `lbfgs`.

**Random Forest:** `n_estimators ∈ {100, 200}`; `max_depth ∈ {None, 40}`; `min_samples_leaf ∈ {1, 2}`; `random_state=42`; `class_weight=None`.

La búsqueda exhaustiva usa `GridSearchCV`, estratificación de cinco folds (`StratifiedKFold`, shuffle, semilla 42) y F1 macro como criterio de selección. Se registran accuracy, precision macro, recall macro y F1 macro en CV; se calcula además ROC-AUC macro one-vs-rest en el conjunto de prueba cuando están disponibles las probabilidades.

## Resultados y MLflow

La ejecución satisfactoria del 17 de octubre de 2026 generó las siguientes métricas agregadas:

| Métrica | Regresión Logística | Random Forest |
|---|---:|---:|
| F1 macro medio en CV | 0.874979 | 0.843913 |
| Accuracy en prueba | 0.887792 | 0.862857 |
| Precision macro | 0.891960 | 0.868230 |
| Recall macro | 0.887792 | 0.862857 |
| F1 macro en prueba | 0.887280 | 0.861592 |
| ROC-AUC macro OVR | 0.997284 | 0.994047 |

Regresión Logística supera a Random Forest en todas las métricas mostradas. La prueba exacta de McNemar reportó 90 ejemplos en los que solo Regresión Logística acertó y 42 en los que solo Random Forest acertó (`p = 0.0000357`, alpha = 0.05); esto indica una diferencia significativa en los aciertos pareados del conjunto de prueba. En contraste, Wilcoxon sobre los cinco folds obtuvo `p = 0.0625`, superior a 0.05, y se considera exploratorio por el número pequeño de folds y la selección de hiperparámetros sobre esas mismas particiones.

Los resultados están resumidos en `results/model_comparison.csv`, `results/statistical_comparison.json` y `results/RESULTADOS_REALES.md`. En la ejecución del usuario, MLflow utiliza una URI SQLite con `mlflow.db`; la base de datos y las capturas de pantalla auténticas se mantienen en el entorno de Colab/Drive y deben copiarse al repositorio si se desean adjuntar. El resumen local no sustituye las tablas detalladas por clase, matrices, salidas de Grid Search y artefactos del entorno original.

## Conclusiones

Con la evidencia agregada disponible, Regresión Logística es el candidato preferido para la siguiente fase de validación. Su F1 macro en prueba fue 0.887280 frente a 0.861592 de Random Forest, y su accuracy fue 0.887792 frente a 0.862857. McNemar respalda una diferencia significativa en el conjunto de prueba, aunque Wilcoxon en validación cruzada no alcanzó significancia al nivel de 0.05. No se concluye superioridad operativa ni aptitud para producción a partir de estas métricas únicamente.

Antes de un piloto bancario se requiere revisar errores por intención, latencia y coste por ejecución, validar con tickets reales mexicanos autorizados y anonimizados, calibrar probabilidades, definir categorías que siempre pasan a revisión humana y monitorear deriva de datos y desempeño. Las métricas de la Actividad 4 siguen siendo contexto histórico y no se mezclan con las de esta actividad.

## Evidencias solicitadas

La ejecución de entrenamiento ya finalizó. Para completar las evidencias visuales de entrega:
1. Abre el panel MLflow usando la URI SQLite de la ejecución.
2. Captura la tabla del experimento con ambas ejecuciones y las métricas.
3. Captura la ejecución de Regresión Logística y la de Random Forest mostrando parámetros, métricas y artefactos.
4. Guarda capturas auténticas en `evidencias/mlflow/` como `01_lista_experimentos.png`, `02_regresion_logistica.png` y `03_random_forest.png`.
5. Conserva las matrices de confusión, reportes por intención, predicciones y resultados detallados de Grid Search generados en tu Drive; cópialos a `results/` antes de publicar si deseas distribuir esos artefactos.

El paquete incluye las métricas agregadas reportadas y sus gráficas derivadas, pero no incluye capturas ficticias ni la base `mlflow.db` que está en el Drive del usuario.

## Fuentes principales

- Material de curso: *Tema 11. Gestión de datos para entrenamiento* y *Tema 12. Entrenamiento y ajuste de modelos* (Tecmilenio, 2026).
- Casanueva et al. (2020), artículo original de BANKING77 / detección de intenciones.
- Documentación oficial de MLflow Tracking y scikit-learn `GridSearchCV`.

Consultar `docs/REPORTE_TECNICO_ACTIVIDAD5.pdf`, `docs/Reporte_Tecnico_Actividad5_Resultados.docx` y `docs/REFERENCIAS_APA.md` para la documentación ampliada.
