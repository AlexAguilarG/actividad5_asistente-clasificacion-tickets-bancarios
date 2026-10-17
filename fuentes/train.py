"""Entrena y compara Regresión Logística y Random Forest con embeddings multilingües.

Ejecutar desde cualquier directorio: python fuentes/train.py
"""
from __future__ import annotations
import os, sys, json, time, hashlib, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, make_scorer)
from scipy.stats import binomtest, wilcoxon
from joblib import dump

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datos" / "datos_limp"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "modelos"
DATASET_ID = "DeepPavlov/banking77-translated"
DATASET_REVISION = "48246f6c99b06096d9dbd7fcbaf273c6d82df2f0"
ENCODER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42
N_SPLITS = 5
SCORING = {
    "accuracy": "accuracy",
    "precision_macro": make_scorer(precision_score, labels=list(range(77)), average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, labels=list(range(77)), average="macro", zero_division=0),
    "f1_macro": make_scorer(f1_score, labels=list(range(77)), average="macro", zero_division=0),
}
INTENT_MAP_PATH = ROOT / "datos" / "diccionario_intenciones.csv"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y_true, y_pred, y_proba, class_ids):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)),
    }
    try:
        result["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_proba, labels=class_ids, multi_class="ovr", average="macro"))
    except Exception as exc:
        warnings.warn(f"No se pudo calcular ROC-AUC macro OVR: {exc}")
        result["roc_auc_ovr_macro"] = float("nan")
    return result


def get_label_names(class_ids):
    if INTENT_MAP_PATH.exists():
        dm = pd.read_csv(INTENT_MAP_PATH)
        mapping = dict(zip(dm["label_id"].astype(int), dm["intencion"].astype(str)))
        return [mapping.get(int(i), f"intent_{i}") for i in class_ids]
    return [f"intent_{i}" for i in class_ids]


def build_searches():
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    searches = {
        "logistic_regression": GridSearchCV(
            LogisticRegression(max_iter=2500, solver="lbfgs", random_state=SEED),
            param_grid={"C": [0.1, 1.0, 10.0], "class_weight": [None, "balanced"]},
            scoring=SCORING, refit="f1_macro", cv=cv, n_jobs=1, return_train_score=True, verbose=1,
        ),
        "random_forest": GridSearchCV(
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            param_grid={"n_estimators": [100, 200], "max_depth": [None, 40], "min_samples_leaf": [1, 2]},
            scoring=SCORING, refit="f1_macro", cv=cv, n_jobs=1, return_train_score=True, verbose=1,
        ),
    }
    return cv, searches


def plot_confusion(cm, labels, path, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Número de ejemplos")
    fig.text(0.5, 0.015, "Los ejes omiten los 77 nombres para mantener legibilidad; consulta el CSV asociado.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0,0.04,1,1])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train_limp.csv"
    test_path = DATA_DIR / "test_limp.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("No se encontraron datos preparados. Ejecuta primero: python fuentes/datos_prep.py")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    for df, nm in [(train_df,"train"),(test_df,"test")]:
        if not {"text","label"}.issubset(df.columns):
            raise ValueError(f"El archivo {nm} debe contener columnas text y label.")
        if df["text"].isna().any() or df["label"].isna().any():
            raise ValueError(f"El archivo {nm} contiene nulos; ejecuta datos_prep.py de nuevo.")
    X_text_train = train_df["text"].astype(str).tolist()
    X_text_test = test_df["text"].astype(str).tolist()
    y_train = train_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()
    class_ids = np.arange(77)
    label_names = get_label_names(class_ids)
    encoder_device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Cargando encoder: {ENCODER_ID} en {encoder_device}")
    encoder = SentenceTransformer(ENCODER_ID, device=encoder_device)
    start = time.perf_counter()
    X_train = encoder.encode(X_text_train, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    X_test = encoder.encode(X_text_test, batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
    embedding_seconds = time.perf_counter() - start
    np.save(DATA_DIR / "X_train_embeddings.npy", X_train)
    np.save(DATA_DIR / "X_test_embeddings.npy", X_test)
    train_hash, test_hash = sha256(train_path), sha256(test_path)
    # MLflow reciente rechaza el backend legacy FileStore por defecto.
    # Usamos SQLite para guardar parámetros, métricas, etiquetas y ejecuciones.
    default_db = (ROOT / "mlflow.db").resolve()
    requested_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()

    if (not requested_uri
            or requested_uri.startswith("file:")
            or requested_uri in {"./mlruns", "mlruns"}):
        mlflow_uri = f"sqlite:///{default_db.as_posix()}"
    elif requested_uri.startswith("sqlite:") or "://" in requested_uri:
        # Permite conservar una URI SQLite o un servidor remoto configurado explícitamente.
        mlflow_uri = requested_uri
    else:
        # Si se proporciona una ruta sin esquema, se interpreta como archivo SQLite.
        db_path = Path(requested_uri).expanduser()
        if not db_path.is_absolute():
            db_path = ROOT / db_path
        mlflow_uri = f"sqlite:///{db_path.resolve().as_posix()}"

    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = "Actividad5_Banking77_IntentClassifier"
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        artifact_root = (ROOT / "mlartifacts").resolve().as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=artifact_root)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    cv, searches = build_searches()
    all_results=[]; predictions={}; cv_fold_scores={}; best_estimators={}
    for model_name, search in searches.items():
        print(f"\n=== Grid Search: {model_name} ===")
        t0=time.perf_counter()
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tags({
                "algorithm": model_name,
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "embedding_model": ENCODER_ID,
                "task": "77-class intent classification",
                "split_policy": "fixed official train/test partitions, balanced class caps, seed=42",
            })
            mlflow.log_params({
                "algorithm": model_name, "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
                "embedding_model": ENCODER_ID, "embedding_dimension": int(X_train.shape[1]),
                "seed": SEED, "cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True)",
                "train_rows": len(y_train), "test_rows": len(y_test), "train_sha256": train_hash,
                "test_sha256": test_hash, "encoder_device": encoder_device,
                "scoring_refit": "f1_macro", "grid_candidates": len(list(ParameterGrid(search.param_grid))),
                "python_version": sys.version.split()[0], "numpy_version": np.__version__,
                "pandas_version": pd.__version__, "sklearn_version": __import__("sklearn").__version__,
                "mlflow_version": mlflow.__version__,
            })
            search.fit(X_train, y_train)
            fit_seconds=time.perf_counter()-t0
            best=search.best_estimator_
            y_pred=best.predict(X_test)
            y_proba=best.predict_proba(X_test)
            met=metrics(y_test,y_pred,y_proba,class_ids)
            best_params={k:(str(v) if v is None else v) for k,v in search.best_params_.items()}
            split_keys=[f"split{i}_test_f1_macro" for i in range(N_SPLITS)]
            folds=[float(search.cv_results_[k][search.best_index_]) for k in split_keys]
            cv_fold_scores[model_name]=folds
            best_estimators[model_name]=best
            mlflow.log_params({f"best_{k}": str(v) for k,v in best_params.items()})
            logged_metrics = {**met,"cv_best_f1_macro_mean":float(search.best_score_),
                "cv_f1_macro_std":float(search.cv_results_["std_test_f1_macro"][search.best_index_]),
                "embedding_seconds":float(embedding_seconds),"search_fit_seconds":float(fit_seconds)}
            mlflow.log_metrics({k: float(v) for k, v in logged_metrics.items() if np.isfinite(float(v))})
            mlflow.log_dict({"best_params":best_params,"cv_fold_f1_macro":folds,"metrics":met},"summary.json")
            cv_df=pd.DataFrame(search.cv_results_).sort_values("rank_test_f1_macro")
            cv_path=RESULTS_DIR/f"grid_results_{model_name}.csv"; cv_df.to_csv(cv_path,index=False)
            mlflow.log_artifact(str(cv_path),artifact_path="grid_search")
            report=classification_report(y_test,y_pred,labels=class_ids,target_names=label_names,output_dict=True,zero_division=0)
            report_df=pd.DataFrame(report).transpose().reset_index().rename(columns={"index":"label_or_average"})
            report_path=RESULTS_DIR/f"classification_report_{model_name}.csv"; report_df.to_csv(report_path,index=False)
            mlflow.log_artifact(str(report_path),artifact_path="evaluation")
            cm=confusion_matrix(y_test,y_pred,labels=class_ids)
            cm_df=pd.DataFrame(cm,index=label_names,columns=label_names)
            cm_path=RESULTS_DIR/f"confusion_matrix_{model_name}.csv"; cm_df.to_csv(cm_path,index_label="true_intent")
            mlflow.log_artifact(str(cm_path),artifact_path="evaluation")
            cm_plot=RESULTS_DIR/f"confusion_matrix_{model_name}.png"
            plot_confusion(cm,label_names,cm_plot,f"Matriz de confusión - {model_name}")
            mlflow.log_artifact(str(cm_plot),artifact_path="evaluation")
            pred_path=RESULTS_DIR/f"predictions_{model_name}.csv"
            pd.DataFrame({"text":X_text_test,"label_true":y_test,"intent_true":np.array(label_names)[y_test],
                          "label_pred":y_pred,"intent_pred":np.array(label_names)[y_pred],
                          "confidence":np.max(y_proba,axis=1)}).to_csv(pred_path,index=False,encoding="utf-8")
            mlflow.log_artifact(str(pred_path),artifact_path="evaluation")
            model_path=MODEL_DIR/f"{model_name}_classifier.joblib"
            dump({"classifier":best,"encoder_id":ENCODER_ID,"label_names":label_names,
                  "dataset_id":DATASET_ID,"dataset_revision":DATASET_REVISION},model_path)
            mlflow.log_artifact(str(model_path),artifact_path="model_bundle")
            # MLflow sklearn flavor saves the estimator that consumes embeddings; encoder ID is logged above.
            try:
                mlflow.sklearn.log_model(best, artifact_path="sklearn_estimator")
            except Exception as exc:
                mlflow.log_param("sklearn_model_logging_warning", str(exc)[:200])
                warnings.warn(f"No se pudo serializar el estimator con flavor sklearn: {exc}")
            all_results.append({"model":model_name,"best_params":json.dumps(best_params,sort_keys=True),
                "cv_f1_macro_mean":float(search.best_score_),
                "cv_f1_macro_std":float(search.cv_results_["std_test_f1_macro"][search.best_index_]),
                "cv_f1_macro_fold_scores":json.dumps(folds),**met,"fit_search_seconds":float(fit_seconds),
                "embedding_seconds":float(embedding_seconds),"run_id":run.info.run_id})
            predictions[model_name]=y_pred
            print({"model":model_name,"best_params":best_params,"metrics":met,"cv_f1":folds})

    comparison=pd.DataFrame(all_results).sort_values("f1_macro",ascending=False)
    cmp_path=RESULTS_DIR/"model_comparison.csv"; comparison.to_csv(cmp_path,index=False)
    # Paired significance tests on a common held-out test set and CV folds.
    names=list(predictions)
    if len(names)>=2:
        a,b=names[0],names[1]
        ca=(predictions[a]==y_test); cb=(predictions[b]==y_test)
        b_only=int(np.sum(ca & ~cb)); c_only=int(np.sum(~ca & cb))
        discordant=b_only+c_only
        mcnemar_p=float(binomtest(min(b_only,c_only),n=discordant,p=0.5,alternative="two-sided").pvalue) if discordant else 1.0
        try:
            w=wilcoxon(cv_fold_scores[a],cv_fold_scores[b],alternative="two-sided",zero_method="wilcox",method="auto")
            wilcoxon_stat=float(w.statistic); wilcoxon_p=float(w.pvalue)
        except Exception:
            wilcoxon_stat=float("nan"); wilcoxon_p=float("nan")
        stat={"model_a":a,"model_b":b,"paired_test":"Exact McNemar test on held-out accuracy errors",
            "model_a_correct_model_b_wrong":b_only,"model_a_wrong_model_b_correct":c_only,
            "mcnemar_exact_p_value":mcnemar_p,"alpha":0.05,
            "mcnemar_interpretation":"significant difference" if mcnemar_p<0.05 else "no statistically significant difference detected",
            "cv_test":"paired Wilcoxon signed-rank test on 5 best-grid fold scores; exploratory due to small number of folds",
            "wilcoxon_statistic":wilcoxon_stat,"wilcoxon_p_value":wilcoxon_p,
            "caution":"The same CV folds were used to select hyperparameters; treat CV significance as exploratory. McNemar compares paired test correctness only and does not establish operational superiority."}
    else:
        stat={"error":"Se requieren dos modelos para la comparación estadística."}
    stat_path=RESULTS_DIR/"statistical_comparison.json"
    stat_path.write_text(json.dumps(stat,ensure_ascii=False,indent=2),encoding="utf-8")
    # Visual comparison: main standardized metrics and CV F1 macro.
    plot_cols=["accuracy","precision_macro","recall_macro","f1_macro","roc_auc_ovr_macro"]
    ax=comparison.set_index("model")[plot_cols].plot(kind="bar",figsize=(12,6),ylim=(0,1),rot=0)
    ax.set_title("Actividad 5: comparación en conjunto de prueba")
    ax.set_ylabel("Métrica (0-1)"); ax.set_xlabel("Algoritmo"); ax.legend(loc="lower right")
    plt.tight_layout(); plot_path=RESULTS_DIR/"model_comparison.png"; plt.savefig(plot_path,dpi=170); plt.close()
    fig,ax=plt.subplots(figsize=(8,5))
    for name,folds in cv_fold_scores.items(): ax.plot(range(1,len(folds)+1),folds,marker="o",label=name)
    ax.set_title("F1 macro por fold del mejor candidato de cada búsqueda")
    ax.set_xlabel("Fold de validación"); ax.set_ylabel("F1 macro"); ax.set_ylim(0,1); ax.legend(); ax.grid(alpha=.25)
    plt.tight_layout(); cv_plot=RESULTS_DIR/"cv_f1_by_fold.png"; plt.savefig(cv_plot,dpi=170); plt.close()
    # A single summary run groups comparison-level artifacts in the UI.
    with mlflow.start_run(run_name="comparison_summary"):
        mlflow.set_tags({"role":"comparison_summary","dataset_id":DATASET_ID,"dataset_revision":DATASET_REVISION})
        mlflow.log_artifact(str(cmp_path)); mlflow.log_artifact(str(stat_path)); mlflow.log_artifact(str(plot_path)); mlflow.log_artifact(str(cv_plot))
        if len(comparison): mlflow.log_metric("best_test_f1_macro",float(comparison.iloc[0]["f1_macro"]))
    print("\n=== Comparación final ===")
    print(comparison[["model","cv_f1_macro_mean","accuracy","precision_macro","recall_macro","f1_macro","roc_auc_ovr_macro"]].to_string(index=False))
    print("\nComparación estadística:",json.dumps(stat,ensure_ascii=False,indent=2))
    print(f"Resultados: {RESULTS_DIR}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Para abrir la interfaz MLflow en el mismo entorno: mlflow ui --backend-store-uri '{mlflow.get_tracking_uri()}' --host 127.0.0.1 --port 5000")

if __name__ == "__main__":
    main()
