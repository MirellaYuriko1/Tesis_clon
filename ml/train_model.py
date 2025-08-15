# ml/train_model.py
from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from Scas.configuracion import get_db  # conexión mysql-connector

MODEL_VERSION = "v1"   # mantén sincronizado con tu app
PREGUNTAS = [f"p{i}" for i in range(1, 39)]

SQL_ULTIMO_X_ALUMNO = f"""
SELECT u.id_usuario, c.id_cuestionario, c.edad, c.genero,
       {", ".join("c."+p for p in PREGUNTAS)},
       r.nivel
FROM (
    SELECT c1.*
    FROM cuestionario c1
    JOIN (
        SELECT id_usuario, MAX(created_at) AS mx
        FROM cuestionario
        GROUP BY id_usuario
    ) ult
      ON ult.id_usuario = c1.id_usuario AND ult.mx = c1.created_at
) c
JOIN usuario u        ON u.id_usuario = c.id_usuario
LEFT JOIN resultado r ON r.id_cuestionario = c.id_cuestionario
"""

def norm_label(s: str) -> str:
    s = (s or "").strip().lower()
    if "muy" in s:
        return "Muy alto"
    if s == "alto":
        return "Alto"
    if "elev" in s:
        return "Elevado"
    return "Normal"

def main():
    print("[ML] Iniciando entrenamiento…")

    # 1) Datos
    print("[ML] Conectando a BD…")
    cn = get_db()
    cur = cn.cursor(dictionary=True)
    cur.execute(SQL_ULTIMO_X_ALUMNO)
    rows = cur.fetchall()
    cur.close(); cn.close()

    df = pd.DataFrame(rows)
    print(f"[ML] Registros leídos: {len(df)}")

    # filtra solo filas con etiqueta
    df = df.dropna(subset=["nivel"]).copy()
    df["nivel_norm"] = df["nivel"].map(norm_label)

    print("[ML] Distribución de clases:")
    print(df["nivel_norm"].value_counts().sort_index())

    feature_cols_num = PREGUNTAS + ["edad"]
    feature_cols_cat = ["genero"]
    X = df[feature_cols_num + feature_cols_cat].copy()
    y = df["nivel_norm"].copy()

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    clf = Pipeline(steps=[("preprocessor", pre), ("model", model)])

    # 2) Validación cruzada (k=5, estratificada)
    print("\n=== Validación Cruzada (5-fold, macro-F1) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
    cv_acc = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    print(f"F1_macro CV: mean={cv_f1.mean():.3f}  std={cv_f1.std():.3f}")
    print(f"Accuracy  CV: mean={cv_acc.mean():.3f}  std={cv_acc.std():.3f}")

    # 3) Hold-out final (20%)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    print("\n=== Reporte (test hold-out) ===")
    print(classification_report(yte, ypred))

    cm = confusion_matrix(yte, ypred, labels=sorted(y.unique()))
    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")

    # 4) Guardar modelo y métricas
    outdir = Path(__file__).parent / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model_v1.joblib"
    joblib.dump(clf, model_path)

    metrics = {
        "model_version": MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_registros": int(len(df)),
        "clases": sorted(y.unique()),
        "cv": {
            "n_splits": 5,
            "f1_macro_mean": float(cv_f1.mean()),
            "f1_macro_std": float(cv_f1.std()),
            "accuracy_mean": float(cv_acc.mean()),
            "accuracy_std": float(cv_acc.std()),
        },
        "holdout": {
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "classification_report": classification_report(yte, ypred, output_dict=True),
            "confusion_matrix": cm.tolist(),
            "labels_order": sorted(y.unique()),
            "n_test": int(len(yte)),
        },
    }
    with open(outdir / "metrics_v1.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelo guardado en: {model_path.resolve()}")
    print(f"📝 Métricas guardadas en: {(outdir / 'metrics_v1.json').resolve()}")

if __name__ == "__main__":
    main()
