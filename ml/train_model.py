# ml/train_model.py
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, cross_val_predict
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from Scas.configuracion import get_db  # conexión mysql-connector

MODEL_VERSION = "v1"   # mantén sincronizado con tu app
PREGUNTAS = [f"p{i}" for i in range(1, 39)]
CLASS_ORDER = ["Normal", "Alto", "Elevado", "Muy alto"]  # orden estable en reportes

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

def _bytes_to_int(x):
    """Convierte posibles bytes/bytearray de MySQL (p1..p38) a int."""
    if isinstance(x, (bytes, bytearray)):
        return int(x[0])
    return x

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

    # ---------------------- 1) Datos ----------------------
    print("[ML] Conectando a BD…")
    cn = get_db()
    cur = cn.cursor(dictionary=True)
    cur.execute(SQL_ULTIMO_X_ALUMNO)
    rows = cur.fetchall()
    cur.close(); cn.close()

    df = pd.DataFrame(rows)
    print(f"[ML] Registros leídos: {len(df)}")

    # Etiqueta normalizada
    df = df.dropna(subset=["nivel"]).copy()
    df["nivel_norm"] = df["nivel"].map(norm_label)

    # Limpieza p1..p38 si vienen como bytes
    pcols = [c for c in df.columns if c.startswith("p")]
    if pcols:
        # evita FutureWarning de applymap usando map por columna
        df[pcols] = df[pcols].apply(lambda col: col.map(_bytes_to_int))

    # Edad numérica
    df["edad"] = pd.to_numeric(df.get("edad"), errors="coerce")
    if df["edad"].isna().all():
        df["edad"] = 13
    else:
        df["edad"] = df["edad"].fillna(df["edad"].median())

    # Género missing → 'Desconocido'
    df["genero"] = df.get("genero").fillna("Desconocido")

    print("[ML] Distribución de clases:")
    print(df["nivel_norm"].value_counts().sort_index())

    # ---------------------- 2) Features -------------------
    feature_cols_num = PREGUNTAS + ["edad"]
    feature_cols_cat = ["genero"]
    X = df[feature_cols_num + feature_cols_cat].copy()
    y = df["nivel_norm"].copy()

    # Orden de clases para reportes
    clases_presentes = list(y.unique())
    classes_sorted = [c for c in CLASS_ORDER if c in clases_presentes] + \
                     [c for c in clases_presentes if c not in CLASS_ORDER]

    # Preprocesamiento:
    # Dense para evitar incompatibilidades con SVM/KNN
    pre_tree = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), feature_cols_cat),
        ]
    )
    pre_scaled = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), feature_cols_cat),
        ]
    )

    # ---------------------- 3) Modelos --------------------
    models = {
        "rf": Pipeline(steps=[
            ("pre", pre_tree),
            ("model", RandomForestClassifier(
                n_estimators=400, random_state=42, class_weight="balanced"
            ))
        ]),
        "svm": Pipeline(steps=[
            ("pre", pre_scaled),
            ("model", SVC(
                kernel="rbf", C=2.0, gamma="scale",
                probability=True,                 # para métricas/umbral futuros
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "knn": Pipeline(steps=[
            ("pre", pre_scaled),
            ("model", KNeighborsClassifier(
                n_neighbors=5, weights="distance", metric="minkowski", p=2
            ))
        ]),
    }

    # ---------------------- 4) Comparativa CV -------------
    print("\n=== Comparativa (5-fold estratificado) — métrica principal: F1_macro ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
    }

    resultados = {}
    best_name, best_score = None, -1.0

    for name, pipe in models.items():
        cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring, return_train_score=False)
        resultados[name] = {
            "f1_macro_mean": float(cv["test_f1_macro"].mean()),
            "f1_macro_std":  float(cv["test_f1_macro"].std()),
            "precision_macro_mean": float(cv["test_precision_macro"].mean()),
            "recall_macro_mean":    float(cv["test_recall_macro"].mean()),
            "accuracy_mean":        float(cv["test_accuracy"].mean()),
            "balanced_accuracy_mean": float(cv["test_balanced_accuracy"].mean()),
            "n": int(len(y)),
        }
        print(f"- {name:5s}  F1_macro={cv['test_f1_macro'].mean():.3f}±{cv['test_f1_macro'].std():.3f}  "
              f"Prec={cv['test_precision_macro'].mean():.3f}  Rec={cv['test_recall_macro'].mean():.3f}  "
              f"Acc={cv['test_accuracy'].mean():.3f}  BalAcc={cv['test_balanced_accuracy'].mean():.3f}")
        if cv["test_f1_macro"].mean() > best_score:
            best_score, best_name = cv["test_f1_macro"].mean(), name

    print(f"\n[ML] Mejor por F1_macro: {best_name}")

    # ---------------------- 5) Hold-out 20% ---------------
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    best_pipe = models[best_name]
    best_pipe.fit(Xtr, ytr)
    ypred = best_pipe.predict(Xte)

    # Métricas globales
    acc = accuracy_score(yte, ypred)
    bal = balanced_accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")
    prec_m = precision_score(yte, ypred, average="macro", zero_division=0)
    rec_m  = recall_score(yte, ypred, average="macro", zero_division=0)

    # Reporte por clase y matriz de confusión
    rep = classification_report(yte, ypred, output_dict=True, zero_division=0)
    cm  = confusion_matrix(yte, ypred, labels=classes_sorted)

    print("\n=== Reporte (test hold-out) ===")
    print(classification_report(yte, ypred, zero_division=0))

    # --------- 6) Subgrupos (GÉNERO) con OOF -------------
    y_pred_cv = cross_val_predict(best_pipe, X, y, cv=skf)

    df_eval = pd.DataFrame({
        "genero": df["genero"],
        "y_true": y.values,
        "y_pred": y_pred_cv
    })

    def _group_report(mask):
        yt = df_eval.loc[mask, "y_true"]
        yp = df_eval.loc[mask, "y_pred"]
        if yt.empty:
            return None
        repg = classification_report(yt, yp, output_dict=True, zero_division=0)
        return {
            "n": int(len(yt)),
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "f1_macro": float(f1_score(yt, yp, average="macro")),
            "classification_report": repg,
            "confusion_matrix": {"labels": classes_sorted, "matrix": confusion_matrix(yt, yp, labels=classes_sorted).tolist()},
        }

    subgroups_genero = {}
    for g in df_eval["genero"].dropna().unique():
        subgroups_genero[g] = _group_report(df_eval["genero"] == g)

    print("\n=== Subgrupos por género (cross-val OOF) ===")
    for g, m in subgroups_genero.items():
        if m:
            print(f"Genero={g}: n={m['n']}, acc={m['accuracy']:.2f}, balAcc={m['balanced_accuracy']:.2f}, f1_macro={m['f1_macro']:.2f}")

    # ---------------------- 7) Guardar --------------------
    outdir = Path(__file__).parent / "models"
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "model_v1.joblib"
    joblib.dump(best_pipe, model_path)

    metrics = {
        "model_version": MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_registros": int(len(df)),
        "clases": classes_sorted,
        "model_selected": best_name,   # 'rf', 'svm' o 'knn'
        "compare": resultados,         # métricas CV de los 3 modelos
        "holdout": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal),
            "precision_macro": float(prec_m),
            "recall_macro": float(rec_m),
            "f1_macro": float(f1m),
            "classification_report": rep,
            "confusion_matrix": cm.tolist(),
            "labels_order": classes_sorted,
            "n_test": int(len(yte)),
        },
        "subgroups": {"genero": subgroups_genero},
    }

    with open(outdir / "metrics_v1.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelo guardado en: {model_path.resolve()}")
    print(f"📝 Métricas guardadas en: {(outdir / 'metrics_v1.json').resolve()}")

if __name__ == "__main__":
    main()
