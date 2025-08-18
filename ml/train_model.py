# ml/train_model.py
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, balanced_accuracy_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression

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
        df[pcols] = df[pcols].applymap(_bytes_to_int)

    # Edad a numérico
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

    # Preprocesamiento
    pre_tree = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols_num),                # árboles no necesitan escalado
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ]
    )
    pre_linear = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
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
        "extratrees": Pipeline(steps=[
            ("pre", pre_tree),
            ("model", ExtraTreesClassifier(
                n_estimators=700, random_state=42, class_weight="balanced", max_features="sqrt"
            ))
        ]),
        "hgb": Pipeline(steps=[
            ("pre", pre_tree),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.1, max_depth=None, max_iter=300, random_state=42
            ))
        ]),
        "logreg": Pipeline(steps=[
            ("pre", pre_linear),
            ("model", LogisticRegression(
                max_iter=2000, multi_class="auto", class_weight="balanced", random_state=42
            ))
        ]),
    }

    # ---------------------- 4) Comparativa CV -------------
    print("\n=== Comparativa (5-fold estratificado) — métrica principal: F1_macro ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {}
    best_name, best_score = None, -1.0

    for name, pipe in models.items():
        cv_f1  = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")
        cv_acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        cv_bal = cross_val_score(pipe, X, y, cv=skf, scoring="balanced_accuracy")
        resultados[name] = {
            "f1_macro_mean": float(cv_f1.mean()),
            "f1_macro_std":  float(cv_f1.std()),
            "accuracy_mean": float(cv_acc.mean()),
            "accuracy_std":  float(cv_acc.std()),
            "balanced_accuracy_mean": float(cv_bal.mean()),
            "balanced_accuracy_std":  float(cv_bal.std()),
            "n": int(len(y)),
        }
        print(f"- {name:10s} F1_macro={cv_f1.mean():.3f}±{cv_f1.std():.3f}  "
              f"Acc={cv_acc.mean():.3f}  BalAcc={cv_bal.mean():.3f}")
        if cv_f1.mean() > best_score:
            best_score, best_name = cv_f1.mean(), name

    print(f"\n[ML] Mejor por F1_macro: {best_name}")

    # ---------------------- 5) Hold-out 20% ---------------
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    best_pipe = models[best_name]
    best_pipe.fit(Xtr, ytr)
    ypred = best_pipe.predict(Xte)

    print("\n=== Reporte (test hold-out) ===")
    print(classification_report(yte, ypred))

    cm  = confusion_matrix(yte, ypred, labels=classes_sorted)
    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")
    bal = balanced_accuracy_score(yte, ypred)

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
        rep  = classification_report(yt, yp, output_dict=True, zero_division=0)
        accg = accuracy_score(yt, yp)
        f1g  = f1_score(yt, yp, average="macro")
        cmg  = confusion_matrix(yt, yp, labels=classes_sorted).tolist()
        return {
            "n": int(len(yt)),
            "accuracy": float(accg),
            "f1_macro": float(f1g),
            "classification_report": rep,
            "confusion_matrix": {"labels": classes_sorted, "matrix": cmg},
        }

    subgroups_genero = {}
    for g in df_eval["genero"].dropna().unique():
        subgroups_genero[g] = _group_report(df_eval["genero"] == g)

    print("\n=== Subgrupos por género (cross-val OOF) ===")
    for g, m in subgroups_genero.items():
        if m:
            print(f"Genero={g}: n={m['n']}, acc={m['accuracy']:.2f}, f1_macro={m['f1_macro']:.2f}")

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
        "model_selected": best_name,
        "compare": resultados,
        "holdout": {
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "balanced_accuracy": float(bal),
            "classification_report": classification_report(yte, ypred, output_dict=True),
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
