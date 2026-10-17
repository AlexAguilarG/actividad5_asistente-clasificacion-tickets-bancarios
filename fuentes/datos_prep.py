"""Descarga, valida y prepara BANKING77 traducido para Actividad 5.

El conjunto completo no se versiona en Git: se descarga desde Hugging Face en tiempo de ejecución.
"""
from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "DeepPavlov/banking77-translated"
DATASET_CONFIG = "es"
DATASET_REVISION = "48246f6c99b06096d9dbd7fcbaf273c6d82df2f0"
TEXT_FIELD = "text_gemma"
SEED = 42
MAX_TRAIN_PER_CLASS = 80
MAX_TEST_PER_CLASS = 25
EXPECTED_COLUMNS = {"text", "label", "text_gemma", "text_qwen"}
RAW_DIR = ROOT / "datos" / "datos_ini"
CLEAN_DIR = ROOT / "datos" / "datos_limp"
RESULTS_DIR = ROOT / "results"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_split(raw: pd.DataFrame, max_per_class: int, split_name: str):
    """Limpia el texto y la etiqueta sin alterar el contenido semántico.

    Los duplicados se miden y reportan, pero no se eliminan automáticamente para no cambiar
    silenciosamente la distribución del benchmark original. La partición train/test se mantiene.
    """
    required = ["text", "label", "text_gemma", "text_qwen"]
    missing_columns = sorted(set(required) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas en {split_name}: {missing_columns}")
    before = len(raw)
    frame = raw[[TEXT_FIELD, "label"]].rename(columns={TEXT_FIELD: "text"}).copy()
    null_text = int(frame["text"].isna().sum())
    null_label = int(frame["label"].isna().sum())
    frame["text"] = frame["text"].astype("string").str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["text", "label"])
    frame = frame[frame["text"].str.len() > 0]
    frame = frame[frame["label"].between(0, 76)]
    frame["label"] = frame["label"].astype("int64")
    duplicate_text_label = int(frame.duplicated(subset=["text", "label"], keep="first").sum())
    # Deliberadamente no se borran duplicados; se reportan para que la decisión sea trazable.
    sampled_parts = []
    for label, group in frame.groupby("label", sort=True):
        n = min(len(group), max_per_class)
        if n:
            sampled_parts.append(group.sample(n=n, random_state=SEED))
    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else frame.iloc[0:0].copy()
    sampled = sampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    report = {
        "split": split_name,
        "rows_raw": int(before),
        "null_text_before_cleaning": null_text,
        "null_label_before_cleaning": null_label,
        "rows_after_basic_cleaning": int(len(frame)),
        "exact_duplicate_text_label_detected_not_removed": duplicate_text_label,
        "rows_experiment_sample": int(len(sampled)),
        "classes_in_sample": int(sampled["label"].nunique()),
        "max_rows_per_class": int(max_per_class),
    }
    return frame, sampled, report


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {DATASET_ID}, configuración={DATASET_CONFIG}, revisión={DATASET_REVISION}")
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION)
    if "train" not in ds or "test" not in ds:
        raise ValueError(f"Se esperaban splits train y test; recibidos: {list(ds.keys())}")
    raw_frames = {}
    sample_frames = {}
    reports = []
    for split, cap in [("train", MAX_TRAIN_PER_CLASS), ("test", MAX_TEST_PER_CLASS)]:
        raw = pd.DataFrame(ds[split][:])
        missing = EXPECTED_COLUMNS - set(raw.columns)
        if missing:
            raise ValueError(f"El split {split} no contiene las columnas esperadas: {sorted(missing)}")
        raw_path = RAW_DIR / f"{split}_raw.csv"
        raw.to_csv(raw_path, index=False, encoding="utf-8")
        clean_full, sample, report = clean_split(raw, cap, split)
        clean_path = CLEAN_DIR / f"{split}_limp.csv"
        sample.to_csv(clean_path, index=False, encoding="utf-8")
        raw_frames[split] = raw
        sample_frames[split] = sample
        reports.append(report)
        report["raw_file_sha256"] = sha256(raw_path)
        report["prepared_file_sha256"] = sha256(clean_path)
        print(f"{split}: {len(raw)} originales -> {len(sample)} ejemplos para el experimento; clases={sample['label'].nunique()}")

    # Check that each partition has the full label space. We do not merge or re-split source partitions.
    expected = set(range(77))
    for split, df in sample_frames.items():
        labels = set(df["label"].unique().tolist())
        if labels != expected:
            raise ValueError(f"{split}: se esperaban 77 clases; faltan={sorted(expected-labels)}; extra={sorted(labels-expected)}")

    counts = []
    for split, df in sample_frames.items():
        cc = df["label"].value_counts().sort_index()
        for label in range(77):
            counts.append({"split": split, "label": label, "count": int(cc.get(label, 0))})
    pd.DataFrame(counts).to_csv(RESULTS_DIR / "class_distribution.csv", index=False)
    report_path = RESULTS_DIR / "data_quality_report.json"
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "text_field": TEXT_FIELD,
        "seed": SEED,
        "cleaning_policy": "trim whitespace; remove null/empty text and null/out-of-range labels; cast labels to int; detect but do not automatically remove exact duplicate text-label rows; keep source train/test partitions separate; sample up to cap per label per split.",
        "python": sys.version,
        "platform": platform.platform(),
        "splits": reports,
        "all_label_ids_expected": list(range(77)),
    }
    report_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # Data quality / class imbalance visualisation.
    try:
        import matplotlib.pyplot as plt
        counts_df = pd.DataFrame(counts)
        pivot = counts_df.pivot(index="label", columns="split", values="count").fillna(0)
        ax = pivot.plot(kind="bar", figsize=(15, 5), width=0.82)
        ax.set_title("Distribución de clases por partición (muestra de experimento)")
        ax.set_xlabel("ID de intención BANKING77")
        ax.set_ylabel("Número de ejemplos")
        ax.legend(title="Partición")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=160)
        plt.close()
    except Exception as exc:
        print(f"No se pudo generar gráfica de distribución: {exc}")
    print(f"Reporte de calidad: {report_path}")
    print("Preparación finalizada. Los CSV generados se mantienen fuera del control de versiones por defecto.")

if __name__ == "__main__":
    main()
