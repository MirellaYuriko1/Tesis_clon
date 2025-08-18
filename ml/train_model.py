# ml/train_model.py
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    balanced_accuracy_score, log_loss
)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from Scas.configuracion import get_db  # conexión mysql-connector

MODEL_VERSION = "v1"   # mantén sincronizado con tu app
PREGUNTAS = [f"p{i}" for i in range(1, 39)]
CLASS_ORDER = ["Normal", "Alto", "Elevado", "Muy alto"]  # orden fijo

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

def multiclass_brier_score(y_true, proba, classes):
    """
    Brier multiclass: mean over samples of sum_j (p_ij - y_ij)^2
    y_true: array-like labels
    proba:  (n_samples, n_classes)
    classes: list of class labels in columns order of proba
    """
    cls_index = {c:i for i,c in enumerate(classes)}
    Y = np.zeros_like(proba)
    for i, yt in enumerate(y_true):
        Y[i, cls_index[yt]] = 1.0
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))

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

    # Orden de clases estable
    classes_sorted = [c for c in CLASS_ORDER if c in y.unique()]

    # Preprocesamiento: escalar numéricas y OHE para categóricas
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_cols_num),
            # Usamos 'sparse=False' por compatibilidad (en 1.4+ emite warning, pero funciona)
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), feature_cols_cat),
        ]
    )

    # ====== 1) Búsqueda de hiperparámetros (inner CV) ======
    print("\n=== GridSearch SVM (RBF) ===")
    base = Pipeline([
        ("preprocessor", pre),
        ("model", SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=42))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "model__C": [0.3, 1, 3, 10],
        "model__gamma": ["scale", 0.1, 0.03, 0.01],
    }
    gs = GridSearchCV(base, param_grid, scoring="f1_macro", cv=skf, n_jobs=-1)
    gs.fit(X, y)
    best_C = gs.best_params_["model__C"]
    best_gamma = gs.best_params_["model__gamma"]
    print(f"Mejor SVM: C={best_C}, gamma={best_gamma} | F1_macro CV={gs.best_score_:.3f}")

    # ====== 2) SVM + Calibración (probabilidades fiables) ======
    svc = SVC(kernel="rbf", C=best_C, gamma=best_gamma,
              class_weight="balanced", probability=False, random_state=42)
    cal_svc = CalibratedClassifierCV(svc, method="isotonic", cv=5)

    clf = Pipeline(steps=[("preprocessor", pre), ("model", cal_svc)])

    # 3) Validación cruzada con el modelo final
    print("\n=== Validación Cruzada (5-fold) ===")
    cv_f1 = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
    cv_acc = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    cv_bacc = cross_val_score(clf, X, y, cv=skf, scoring="balanced_accuracy")
    print(f"F1_macro CV: mean={cv_f1.mean():.3f}  std={cv_f1.std():.3f}")
    print(f"Accuracy  CV: mean={cv_acc.mean():.3f}  std={cv_acc.std():.3f}")
    print(f"Bal.Acc  CV: mean={cv_bacc.mean():.3f}  std={cv_bacc.std():.3f}")

    # Probabilidades OOF para calibración y subgrupos
    oof_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    y_pred_cv = np.array([classes_sorted[i] for i in np.argmax(oof_proba, axis=1)])

    df_eval = pd.DataFrame({
        "genero": df["genero"].values,
        "y_true": y.values,
        "y_pred": y_pred_cv
    })

    def _group_report(mask):
        yt = df_eval.loc[mask, "y_true"]
        yp = df_eval.loc[mask, "y_pred"]
        if yt.empty:
            return None
        rep = classification_report(yt, yp, output_dict=True, zero_division=0)
        acc_g = accuracy_score(yt, yp)
        f1m_g = f1_score(yt, yp, average="macro")
        cm_g = confusion_matrix(yt, yp, labels=classes_sorted).tolist()
        return {
            "n": int(len(yt)),
            "accuracy": float(acc_g),
            "f1_macro": float(f1m_g),
            "classification_report": rep,
            "confusion_matrix": {"labels": classes_sorted, "matrix": cm_g},
        }

    subgroups_genero = {}
    for g in pd.Series(df_eval["genero"]).dropna().unique():
        subgroups_genero[g] = _group_report(df_eval["genero"] == g)

    print("\n=== Subgrupos por género (OOF) ===")
    for g, m in subgroups_genero.items():
        if m:
            print(f"Genero={g}: n={m['n']}, acc={m['accuracy']:.2f}, f1_macro={m['f1_macro']:.2f}")

    # Calibración OOF
    oof_logloss = log_loss(y, oof_proba, labels=classes_sorted)
    oof_brier = multiclass_brier_score(y.values, oof_proba, classes_sorted)
    print(f"\nCalibración OOF -> LogLoss={oof_logloss:.3f}  Brier={oof_brier:.3f}")

    # 4) Hold-out final (20%)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    yproba_te = clf.predict_proba(Xte)

    print("\n=== Reporte (test hold-out) ===")
    print(classification_report(yte, ypred, zero_division=0))

    cm = confusion_matrix(yte, ypred, labels=classes_sorted)
    acc = accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")
    bacc = balanced_accuracy_score(yte, ypred)
    te_logloss = log_loss(yte, yproba_te, labels=classes_sorted)
    te_brier = multiclass_brier_score(yte.values, yproba_te, classes_sorted)

    # 5) Guardar modelo y métricas
    outdir = Path(__file__).parent / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model_v1.joblib"
    joblib.dump(clf, model_path)

    metrics = {
        "model_version": MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_registros": int(len(df)),
        "clases": classes_sorted,
        "grid_search": {
            "best_params": {"C": float(best_C), "gamma": best_gamma},
            "scoring": "f1_macro"
        },
        "cv": {
            "n_splits": 5,
            "f1_macro_mean": float(cv_f1.mean()),
            "f1_macro_std": float(cv_f1.std()),
            "accuracy_mean": float(cv_acc.mean()),
            "accuracy_std": float(cv_acc.std()),
            "balanced_accuracy_mean": float(cv_bacc.mean()),
            "balanced_accuracy_std": float(cv_bacc.std()),
            "calibration": {
                "log_loss_oof": float(oof_logloss),
                "brier_oof": float(oof_brier)
            }
        },
        "subgroups": {
            "genero": subgroups_genero
        },
        "holdout": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "f1_macro": float(f1m),
            "classification_report": classification_report(yte, ypred, output_dict=True, zero_division=0),
            "confusion_matrix": cm.tolist(),
            "labels_order": classes_sorted,
            "n_test": int(len(yte)),
            "calibration": {
                "log_loss": float(te_logloss),
                "brier": float(te_brier)
            }
        },
    }
    with open(outdir / "metrics_v1.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelo guardado en: {model_path.resolve()}")
    print(f"📝 Métricas guardadas en: {(outdir / 'metrics_v1.json').resolve()}")
    print("ℹ️  Clases (orden):", classes_sorted)

if __name__ == "__main__":
    main()
