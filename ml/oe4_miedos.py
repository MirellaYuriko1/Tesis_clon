# ml/oe4_miedos.py
# ------------------------------------------------------------
# OE4 - Miedos (Dim4)
# Principal: UPPERBOUND (usa SOLO los ítems de Dim4)
# Opcional : LOSO (excluye Dim4 y usa Dim1,2,3,5,6 + edad + género)
# Salidas: reports/oe4_miedos/{tables,figs,figs_ml,tables_ml}
# ------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")  # backend sin GUI

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score, learning_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score,
    balanced_accuracy_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from Scas.configuracion import get_db
from Scas.regla_puntuaciones import DIMENSIONES, DIM_NOMBRES, PRETTY, interpreta_normas

# ================= Config =================
RANDOM_STATE = 42
LEVELS = ["Normal", "Alto", "Elevado", "Muy alto"]
PREGUNTAS = [f"p{i}" for i in range(1, 39)]
DIM_KEY = "Dim4"  # Miedos
OUTDIR = Path(__file__).parent / "reports" / "oe4_miedos"
RUN_LOSO = False   # True si quieres también la variante LOSO

# ======= Textos para títulos y leyendas =======
UI = {
    "pred_title": "Miedos: diagnósticos predichos vs. esperados",
    "pred_sub":   "Comparación por nivel (Normal, Alto, Elevado, Muy alto)",
    "leg_esp":    "Esperado (normas SCAS)",
    "leg_pred":   "Predicción (Random Forest)",
    "x_nivel":    "Nivel",
    "y_n":        "Número de estudiantes",

    "cm_title":        "Matriz de confusión — Miedos (Random Forest)",
    "cm_title_norm":   "Matriz de confusión (porcentaje por fila) — Miedos",
    "color_casos":     "Casos",
    "color_pct":       "% de casos",
    "x_pred":          "Predicho",
    "y_exp":           "Esperado",

    "roc_title": "Curvas ROC por nivel — Miedos",
    "roc_x":     "Falsos positivos (FPR)",
    "roc_y":     "Verdaderos positivos (TPR)",

    "pr_title":  "Curvas Precisión–Recall por nivel — Miedos",
    "pr_x":      "Recall",
    "pr_y":      "Precisión",

    "cal_title": 'Calibración de probabilidades — nivel "Muy alto" (Miedos)',
    "cal_leg1":  "Observado",
    "cal_leg2":  "Perfectamente calibrado",
    "cal_x":     "Probabilidad predicha",
    "cal_y":     "Fracción observada",

    "cv_title":  "Estabilidad en validación cruzada (k=5) — F1_macro (Miedos)",
    "cv_y":      "F1_macro",

    "thr_title": 'Ajuste de umbral — nivel "Muy alto" (Miedos)',
    "thr_x":     "Umbral de decisión",
    "thr_y":     "Valor",

    "imp_items_title": "Aporte de cada ítem (Dim4) — Miedos",
    "imp_items_x":     "Importancia (permutación)",
    "imp_items_y":     "Ítems de la subescala",

    "learn_title": "Curva de aprendizaje — Random Forest (F1_macro) — Miedos",
    "learn_x":     "Tamaño del conjunto de entrenamiento",
    "learn_y":     "F1_macro",
}

# ================ Helpers =================
def _bytes_to_int(x):
    if isinstance(x, (bytes, bytearray)):
        return int(x[0])
    return x

def _brier_multiclass(y_true, proba, classes):
    Y = np.zeros_like(proba)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, y in enumerate(y_true):
        Y[i, class_to_idx[y]] = 1.0
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))

def _normaliza_genero(g):
    s = (str(g or "")).strip().lower()
    if s in ("f", "fem", "femenino", "female"):
        return "Femenino"
    if s in ("m", "masc", "masculino", "male"):
        return "Masculino"
    return str(g or "")

def _pred_vs_esp_table(y_true, y_pred):
    ct_true = pd.Series(y_true).value_counts().reindex(LEVELS, fill_value=0)
    ct_pred = pd.Series(y_pred).value_counts().reindex(LEVELS, fill_value=0)
    return pd.DataFrame({"Nivel": LEVELS, "Esperados": ct_true.values, "Predichos": ct_pred.values})

def ovr_counts_and_metrics(y_true, y_pred, labels):
    filas = []
    y_true = pd.Series(y_true); y_pred = pd.Series(y_pred)
    for cls in labels:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        tn = int(((y_true != cls) & (y_pred != cls)).sum())
        acc = (tp + tn) / max(tn + fp + fn + tp, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 0.0 if (prec + rec) == 0 else 2 * (prec * rec) / (prec + rec)
        filas.append({"Nivel": cls, "f00": tn, "f01": fp, "f10": fn, "f11": tp,
                      "Accuracy": round(acc,3), "Recall": round(rec,3),
                      "Precision": round(prec,3), "F1-score": round(f1,3)})
    return pd.DataFrame(filas)

# ============== Datos & Etiquetas ==============
def cargar_dataset():
    sql = f"""
    SELECT u.id_usuario, c.id_cuestionario, c.edad, c.genero,
           {", ".join("c."+p for p in PREGUNTAS)}
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
    JOIN usuario u ON u.id_usuario = c.id_usuario
    """
    cn = get_db(); cur = cn.cursor(dictionary=True)
    cur.execute(sql); rows = cur.fetchall()
    cur.close(); cn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No hay datos en 'cuestionario'.")

    pcols = [c for c in df.columns if c.startswith("p")]
    df[pcols] = df[pcols].applymap(_bytes_to_int)
    df["genero"] = df["genero"].map(_normaliza_genero)

    for dim, idxs in DIMENSIONES.items():
        df[f"{dim}_sum"] = df[[f"p{i}" for i in idxs]].sum(axis=1)
    df["total_sum"] = df[[f"p{i}" for i in range(1, 39)]].sum(axis=1)

    y_esp = []
    for _, row in df.iterrows():
        sumas_dim = {dim: row[f"{dim}_sum"] for dim in DIMENSIONES.keys()}
        subesc, _tot = interpreta_normas(
            row.get("genero"), int(row.get("edad") or 0),
            sumas_dim, int(row["total_sum"])
        )
        nombre_sub = DIM_NOMBRES[DIM_KEY]  # "Miedos"
        label = subesc.get(nombre_sub)
        y_esp.append(label if label in LEVELS else None)
    df["y_expected"] = y_esp
    df = df.dropna(subset=["y_expected"]).copy()
    return df

# ============== UPPERBOUND (principal) ==============
def evaluar_upper_rf(df):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "figs").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "tables").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "figs_ml").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "tables_ml").mkdir(parents=True, exist_ok=True)

    items_dim = [f"p{i}" for i in DIMENSIONES[DIM_KEY]]
    X = df[items_dim].copy()
    y = df["y_expected"].copy()

    clf = Pipeline([
        ("pre", StandardScaler(with_mean=False)),
        ("mdl", RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_STATE, class_weight="balanced"
        ))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred  = cross_val_predict(clf, X, y, cv=skf, method="predict")
    y_proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")

    # métricas
    f1m = f1_score(y, y_pred, average="macro")
    acc = accuracy_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    try:
        auc_macro = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_macro = None
    brier = _brier_multiclass(y.values, y_proba, LEVELS)
    cm  = confusion_matrix(y, y_pred, labels=LEVELS)
    rep = classification_report(y, y_pred, output_dict=True, zero_division=0)

    # ---------- Tablas ----------
    _pred_vs_esp_table(y, y_pred).to_csv(
        OUTDIR / "tables" / "pred_vs_esp_UpperBound_rf.csv", index=False, encoding="utf-8"
    )
    ovr_counts_and_metrics(y, y_pred, LEVELS).to_csv(
        OUTDIR / "tables" / "tabla12_ovr_UpperBound_rf.csv", index=False, encoding="utf-8"
    )
    filas = []
    for cls in LEVELS:
        d = rep.get(cls, {"precision":0,"recall":0,"f1-score":0,"support":0})
        filas.append({"Clase": cls, "Precision": round(d["precision"],3),
                      "Recall": round(d["recall"],3), "F1": round(d["f1-score"],3),
                      "Soporte": int(d["support"])})
    filas.append({"Clase":"GLOBAL",
                  "Precision": round(rep["macro avg"]["precision"],3),
                  "Recall": round(rep["macro avg"]["recall"],3),
                  "F1": round(rep["macro avg"]["f1-score"],3),
                  "Accuracy": round(acc,3),
                  "BalancedAccuracy": round(bal,3),
                  "AUC_OVR": None if auc_macro is None else round(auc_macro,3),
                  "Brier": round(brier,3),
                  "n": int(len(y))})
    pd.DataFrame(filas).to_csv(OUTDIR / "tables" / "metricas_UpperBound_rf.csv",
                               index=False, encoding="utf-8")

    # ---------- Figuras principales ----------
    import matplotlib.pyplot as plt
    x = np.arange(len(LEVELS))
    ct_true = pd.Series(y).value_counts().reindex(LEVELS, fill_value=0).values
    ct_pred = pd.Series(y_pred).value_counts().reindex(LEVELS, fill_value=0).values

    # líneas
    plt.figure(figsize=(7.2,4.2), dpi=160)
    plt.plot(x, ct_true, marker="o", linewidth=2, label=UI["leg_esp"])
    plt.plot(x, ct_pred, marker="o", linewidth=2, label=UI["leg_pred"])
    for i,v in enumerate(ct_true): plt.text(x[i], v, f"{v}", ha="center", va="bottom", fontsize=9)
    for i,v in enumerate(ct_pred): plt.text(x[i], v, f"{v}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, LEVELS); plt.grid(axis="y", alpha=.25)
    plt.title(f"{UI['pred_title']}\n{UI['pred_sub']}")
    plt.xlabel(UI["x_nivel"]); plt.ylabel(UI["y_n"]); plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "figs" / "pred_vs_esp_UpperBound_rf.png"); plt.close()

    # barras
    width = 0.38
    plt.figure(figsize=(7.2,4.2), dpi=160)
    plt.bar(x-width/2, ct_true, width, label=UI["leg_esp"])
    plt.bar(x+width/2, ct_pred, width, label=UI["leg_pred"])
    for i,v in enumerate(ct_true): plt.text(x[i]-width/2, v, f"{v}", ha="center", va="bottom", fontsize=9)
    for i,v in enumerate(ct_pred): plt.text(x[i]+width/2, v, f"{v}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, LEVELS); plt.grid(axis="y", alpha=.25)
    plt.title(f"{UI['pred_title']}\n{UI['pred_sub']}")
    plt.xlabel(UI["x_nivel"]); plt.ylabel(UI["y_n"]); plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "figs" / "pred_vs_esp_barras_UpperBound_rf.png"); plt.close()

    # confusión
    plt.figure(figsize=(5.6,4.6), dpi=160)
    im = plt.imshow(cm, aspect="auto")
    plt.title(UI["cm_title"])
    plt.xlabel(UI["x_pred"]); plt.ylabel(UI["y_exp"])
    plt.xticks(range(len(LEVELS)), LEVELS); plt.yticks(range(len(LEVELS)), LEVELS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j,i,int(cm[i,j]),ha="center",va="center",fontsize=9)
    cbar = plt.colorbar(im); cbar.set_label(UI["color_casos"])
    plt.tight_layout(); plt.savefig(OUTDIR / "figs" / "confusion_UpperBound_rf.png"); plt.close()

    # ---------- Figuras ML adicionales ----------
    ml_dir = OUTDIR / "figs_ml"

    # ROC OVR
    Y_bin = label_binarize(y, classes=LEVELS)
    plt.figure(figsize=(6.2,4.6), dpi=160)
    for i, cls in enumerate(LEVELS):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],"k--", linewidth=1)
    plt.xlabel(UI["roc_x"]); plt.ylabel(UI["roc_y"]); plt.title(UI["roc_title"])
    plt.legend(); plt.tight_layout(); plt.savefig(ml_dir / "roc_ovr_UpperBound_rf.png"); plt.close()

    # Precision–Recall OVR
    plt.figure(figsize=(6.2,4.6), dpi=160)
    for i, cls in enumerate(LEVELS):
        precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(Y_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, label=f"{cls} (AP={ap:.2f})")
    plt.xlabel(UI["pr_x"]); plt.ylabel(UI["pr_y"]); plt.title(UI["pr_title"])
    plt.legend(); plt.tight_layout(); plt.savefig(ml_dir / "pr_ovr_UpperBound_rf.png"); plt.close()

    # Calibración para 'Muy alto'
    idx_muy = LEVELS.index("Muy alto")
    prob_muy = y_proba[:, idx_muy]
    y_muy = (np.array(y) == "Muy alto").astype(int)
    frac_pos, mean_pred = calibration_curve(y_muy, prob_muy, n_bins=8, strategy="uniform")
    plt.figure(figsize=(6.2,4.6), dpi=160)
    plt.plot(mean_pred, frac_pos, marker="o", label=UI["cal_leg1"])
    plt.plot([0,1],[0,1],"k--", label=UI["cal_leg2"])
    plt.xlabel(UI["cal_x"]); plt.ylabel(UI["cal_y"]); plt.title(UI["cal_title"])
    plt.legend(); plt.tight_layout(); plt.savefig(ml_dir / "calibracion_muy_alto_UpperBound_rf.png"); plt.close()

    # Confusión normalizada
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(5.8,4.8), dpi=160)
    im = plt.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    plt.title(UI["cm_title_norm"])
    plt.xlabel(UI["x_pred"]); plt.ylabel(UI["y_exp"])
    plt.xticks(range(len(LEVELS)), LEVELS); plt.yticks(range(len(LEVELS)), LEVELS)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i,j]*100:.0f}%", ha="center", va="center", fontsize=9)
    cbar = plt.colorbar(im); cbar.set_label(UI["color_pct"])
    plt.tight_layout(); plt.savefig(ml_dir / "confusion_norm_UpperBound_rf.png"); plt.close()

    # Estabilidad CV (F1_macro)
    cv_f1 = cross_val_score(clf, X, y, cv=skf, scoring="f1_macro")
    plt.figure(figsize=(4.8,4.6), dpi=160)
    plt.boxplot(cv_f1, vert=True, labels=["F1_macro"])
    plt.title(f"{UI['cv_title']}\nMedia={cv_f1.mean():.3f}, SD={cv_f1.std():.3f}")
    plt.ylabel(UI["cv_y"])
    plt.tight_layout(); plt.savefig(ml_dir / "cv_boxplot_f1_UpperBound_rf.png"); plt.close()
    pd.DataFrame({"fold": list(range(1,len(cv_f1)+1)), "f1_macro": cv_f1})\
      .to_csv(OUTDIR / "tables_ml" / "cv_f1_folds_UpperBound.csv", index=False, encoding="utf-8")

    # Umbral para 'Muy alto'
    ths = np.linspace(0.1, 0.9, 9)
    precs, recs = [], []
    for t in ths:
        y_hat = np.where(prob_muy >= t, 1, 0)
        tp = ((y_muy == 1) & (y_hat == 1)).sum()
        fp = ((y_muy == 0) & (y_hat == 1)).sum()
        fn = ((y_muy == 1) & (y_hat == 0)).sum()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        precs.append(prec); recs.append(rec)
    import matplotlib.pyplot as plt2
    plt2.figure(figsize=(6.2,4.6), dpi=160)
    plt2.plot(ths, precs, marker="o", label="Precisión")
    plt2.plot(ths, recs, marker="o", label="Recall")
    plt2.xlabel(UI["thr_x"]); plt2.ylabel(UI["thr_y"]); plt2.title(UI["thr_title"])
    plt2.legend(); plt2.tight_layout(); plt2.savefig(ml_dir / "umbral_muy_alto_UpperBound_rf.png"); plt2.close()

    # Importancia por ítem (permutación) — interpretativa
    rf_fit = clf.fit(X, y).named_steps["mdl"]
    r = permutation_importance(rf_fit, X, y, n_repeats=20, random_state=RANDOM_STATE)
    order = np.argsort(r.importances_mean)
    plt2.figure(figsize=(6.8,4.8), dpi=160)
    plt2.barh([items_dim[i] for i in order], [r.importances_mean[i] for i in order])
    plt2.title(UI["imp_items_title"])
    plt2.xlabel(UI["imp_items_x"]); plt2.ylabel(UI["imp_items_y"])
    plt2.tight_layout(); plt2.savefig(ml_dir / "importancias_items_dim_UpperBound_rf.png"); plt2.close()

    # Learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=skf, scoring="f1_macro",
        train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True, random_state=RANDOM_STATE
    )
    plt.figure(figsize=(6.6,4.8), dpi=160)
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Entrenamiento")
    plt.plot(train_sizes, test_scores.mean(axis=1), marker="o", label="Validación (CV)")
    plt.xlabel(UI["learn_x"]); plt.ylabel(UI["learn_y"]); plt.legend()
    plt.title(UI["learn_title"])
    plt.tight_layout(); plt.savefig(ml_dir / "learning_curve_UpperBound_rf.png"); plt.close()

    return {
        "f1_macro": float(f1m),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "auc_ovr_macro": None if auc_macro is None else float(auc_macro),
        "brier": float(brier),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y))
    }

# ============== LOSO (opcional) ==============
def evaluar_loso_rf(df):
    otras_dims = [d for d in DIMENSIONES.keys() if d != DIM_KEY]
    feature_num = [f"{d}_sum" for d in otras_dims] + ["edad"]
    feature_cat = ["genero"]
    X = df[feature_num + feature_cat].copy()
    y = df["y_expected"].copy()

    clf = Pipeline([
        ("pre", ColumnTransformer([
            ("num", Pipeline([("sc", StandardScaler(with_mean=False))]), feature_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cat),
        ])),
        ("mdl", RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_STATE, class_weight="balanced"
        ))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred  = cross_val_predict(clf, X, y, cv=skf, method="predict")

    acc = accuracy_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")

    (OUTDIR / "tables").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "figs").mkdir(parents=True, exist_ok=True)
    _pred_vs_esp_table(y, y_pred).to_csv(
        OUTDIR / "tables" / "pred_vs_esp_LOSO_rf.csv", index=False, encoding="utf-8"
    )
    cm = confusion_matrix(y, y_pred, labels=LEVELS)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.6,4.6), dpi=160)
    im = plt.imshow(cm, aspect="auto")
    plt.title(UI["cm_title"].replace("Random Forest", "Random Forest (LOSO)"))
    plt.xlabel(UI["x_pred"]); plt.ylabel(UI["y_exp"])
    plt.xticks(range(len(LEVELS)), LEVELS); plt.yticks(range(len(LEVELS)), LEVELS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j,i,int(cm[i,j]),ha="center",va="center",fontsize=9)
    cbar = plt.colorbar(im); cbar.set_label(UI["color_casos"])
    plt.tight_layout(); plt.savefig(OUTDIR / "figs" / "confusion_LOSO_rf.png"); plt.close()

    return {"f1_macro": float(f1m), "accuracy": float(acc), "balanced_accuracy": float(bal)}

# ==================== MAIN ====================
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "figs").mkdir(parents=True, exist_ok=True)
    (OUTDIR / "tables").mkdir(parents=True, exist_ok=True)

    df = cargar_dataset()
    if df.empty:
        raise RuntimeError("No hay casos con norma aplicable (edad 12–15 y género válido).")

    res_upper = evaluar_upper_rf(df)
    resumen_final = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_registros_usados": int(len(df)),
        "levels_order": LEVELS,
        "dim_key": DIM_KEY,
        "dim_name": PRETTY[DIM_NOMBRES[DIM_KEY]],
        "upperbound": res_upper,
    }

    if RUN_LOSO:
        res_loso = evaluar_loso_rf(df)
        resumen_final["loso"] = res_loso

    with open(OUTDIR / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen_final, f, ensure_ascii=False, indent=2)

    print("\n✅ Reportes generados en:", OUTDIR.resolve())
    print(f"   UpperBound → F1={res_upper['f1_macro']:.3f}  Acc={res_upper['accuracy']:.3f}  BalAcc={res_upper['balanced_accuracy']:.3f}")
    if RUN_LOSO:
        print(f"   LOSO       → F1={res_loso['f1_macro']:.3f}  Acc={res_loso['accuracy']:.3f}  BalAcc={res_loso['balanced_accuracy']:.3f}")

if __name__ == "__main__":
    main()