# ml/oe_total.py
# ------------------------------------------------------------
# OE (TOTAL) - Modelo sobre puntaje total SCAS
# Usa TODAS las preguntas p1..p38 + edad + genero
# Salidas: reports/oe_total/{figs,figs_ml,tables,tables_ml}
# Importancias: importancias por ítem y agregadas por subescala
# ------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")  # backend sin GUI

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, learning_curve
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score,
    balanced_accuracy_score, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from Scas.configuracion import get_db
from Scas.regla_puntuaciones import DIMENSIONES, DIM_NOMBRES, PRETTY

# ================= Config =================
RANDOM_STATE = 42
LEVELS = ["Normal", "Alto", "Elevado", "Muy alto"]
PREGUNTAS = [f"p{i}" for i in range(1, 39)]
OUTDIR = Path(__file__).parent / "reports" / "oe_total"
(OUTDIR / "figs").mkdir(parents=True, exist_ok=True)
(OUTDIR / "figs_ml").mkdir(parents=True, exist_ok=True)
(OUTDIR / "tables").mkdir(parents=True, exist_ok=True)
(OUTDIR / "tables_ml").mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(__file__).parent / "models" / "model_v1.joblib"  # tu pipeline entrenado

UI = {
    "pred_title": "Nivel TOTAL de ansiedad: predicho vs. esperado",
    "pred_sub":   "Comparación por nivel (Normal, Alto, Elevado, Muy alto)",
    "leg_esp":    "Esperado (normas SCAS)",
    "leg_pred":   "Predicción (modelo total)",
    "x_nivel":    "Nivel",
    "y_n":        "Número de estudiantes",

    "cm_title":        "Matriz de confusión — TOTAL (modelo)",
    "cm_title_norm":   "Matriz de confusión (porcentaje por fila) — TOTAL",
    "color_casos":     "Casos",
    "color_pct":       "% de casos",
    "x_pred":          "Predicho",
    "y_exp":           "Esperado",

    "roc_title": "Curvas ROC por nivel — TOTAL (si hay probabilidades)",
    "roc_x":     "Falsos positivos (FPR)",
    "roc_y":     "Verdaderos positivos (TPR)",

    "pr_title":  "Curvas Precisión–Recall por nivel — TOTAL (si hay probabilidades)",
    "pr_x":      "Recall",
    "pr_y":      "Precisión",

    "cal_title": 'Calibración de probabilidades — nivel "Muy alto" (TOTAL)',
    "cal_leg1":  "Observado",
    "cal_leg2":  "Perfectamente calibrado",
    "cal_x":     "Probabilidad predicha",
    "cal_y":     "Fracción observada",

    "cv_title":  "Estabilidad en validación cruzada (k=5) — F1_macro (TOTAL)",
    "cv_y":      "F1_macro",

    "imp_items_title": "Aporte de cada variable — MODELO TOTAL",
    "imp_items_x":     "Importancia (permutación, F1_macro)",
    "imp_items_y":     "Variables (p1..p38, edad, genero)",

    "imp_dim_title":   "Importancia agregada por subescala — MODELO TOTAL",
    "imp_dim_x":       "Suma de importancias (permutación, F1_macro)",
    "imp_dim_y":       "Grupo (Dimensión / edad / genero)",

    "learn_title": "Curva de aprendizaje — MODELO TOTAL (F1_macro)",
    "learn_x":     "Tamaño del conjunto de entrenamiento",
    "learn_y":     "F1_macro",
}

# ================ Helpers =================
def _bytes_to_int(x):
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

# ============== Datos: último cuestionario por alumno con nivel (regla) ==============
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

def cargar_dataset_total():
    cn = get_db(); cur = cn.cursor(dictionary=True)
    cur.execute(SQL_ULTIMO_X_ALUMNO); rows = cur.fetchall()
    cur.close(); cn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No hay datos en 'cuestionario'.")

    # p1..p38 pueden llegar como bytes; normalizamos
    pcols = [c for c in df.columns if c.startswith("p")]
    df[pcols] = df[pcols].applymap(_bytes_to_int)

    # etiqueta (nivel por reglas) y filtrado
    df = df.dropna(subset=["nivel"]).copy()
    df["nivel_norm"] = df["nivel"].map(norm_label)

    # features exactamente como en train_model.py
    feature_cols = PREGUNTAS + ["edad", "genero"]
    X = df[feature_cols].copy()
    y = df["nivel_norm"].copy()

    return X, y

# ============== Reportes principales ==============
def generar_reportes_total():
    X, y = cargar_dataset_total()
    pipe = load(MODEL_PATH)  # pipeline entrenado (RF, SVM, etc.)

    # === Validación cruzada OOF (k=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(pipe, X, y, cv=skf, method="predict")

    has_proba = hasattr(pipe, "predict_proba")
    y_proba = None
    if has_proba:
        try:
            y_proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")
        except Exception:
            has_proba = False

    # Métricas globales
    f1m = f1_score(y, y_pred, average="macro")
    acc = accuracy_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    auc_macro = None
    if has_proba:
        try:
            auc_macro = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auc_macro = None

    rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
    cm  = confusion_matrix(y, y_pred, labels=LEVELS)

    # ---------- Tablas ----------
    # Predicho vs esperado (conteos)
    ct_true = pd.Series(y).value_counts().reindex(LEVELS, fill_value=0)
    ct_pred = pd.Series(y_pred).value_counts().reindex(LEVELS, fill_value=0)
    pd.DataFrame({"Nivel": LEVELS, "Esperados": ct_true.values, "Predichos": ct_pred.values})\
      .to_csv(OUTDIR / "tables" / "pred_vs_esp_TOTAL.csv", index=False, encoding="utf-8")

    # Métricas por clase + globales
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
                  "n": int(len(y))})
    pd.DataFrame(filas).to_csv(OUTDIR / "tables" / "metricas_TOTAL.csv",
                               index=False, encoding="utf-8")

    # ---------- Figuras: barras y líneas ----------
    x = np.arange(len(LEVELS))
    plt.figure(figsize=(7.2,4.2), dpi=160)
    plt.plot(x, ct_true.values, marker="o", linewidth=2, label=UI["leg_esp"])
    plt.plot(x, ct_pred.values, marker="o", linewidth=2, label=UI["leg_pred"])
    for i,v in enumerate(ct_true.values): plt.text(x[i], v, f"{v}", ha="center", va="bottom", fontsize=9)
    for i,v in enumerate(ct_pred.values): plt.text(x[i], v, f"{v}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, LEVELS); plt.grid(axis="y", alpha=.25)
    plt.title(f"{UI['pred_title']}\n{UI['pred_sub']}")
    plt.xlabel(UI["x_nivel"]); plt.ylabel(UI["y_n"]); plt.legend()
    plt.tight_layout(); plt.savefig(OUTDIR / "figs" / "pred_vs_esp_TOTAL_lineas.png"); plt.close()

    width = 0.38
    plt.figure(figsize=(7.2,4.2), dpi=160)
    plt.bar(x-width/2, ct_true.values, width, label=UI["leg_esp"])
    plt.bar(x+width/2, ct_pred.values, width, label=UI["leg_pred"])
    for i,v in enumerate(ct_true.values): plt.text(x[i]-width/2, v, f"{v}", ha="center", va="bottom", fontsize=9)
    for i,v in enumerate(ct_pred.values): plt.text(x[i]+width/2, v, f"{v}", ha="center", va="bottom", fontsize=9)
    plt.xticks(x, LEVELS); plt.grid(axis="y", alpha=.25)
    plt.title(f"{UI['pred_title']}\n{UI['pred_sub']}")
    plt.xlabel(UI["x_nivel"]); plt.ylabel(UI["y_n"]); plt.legend()
    plt.tight_layout(); plt.savefig(OUTDIR / "figs" / "pred_vs_esp_TOTAL_barras.png"); plt.close()

    # ---------- Confusión ----------
    plt.figure(figsize=(5.6,4.6), dpi=160)
    im = plt.imshow(cm, aspect="auto")
    plt.title(UI["cm_title"])
    plt.xlabel(UI["x_pred"]); plt.ylabel(UI["y_exp"])
    plt.xticks(range(len(LEVELS)), LEVELS); plt.yticks(range(len(LEVELS)), LEVELS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): plt.text(j,i,int(cm[i,j]),ha="center",va="center",fontsize=9)
    cbar = plt.colorbar(im); cbar.set_label(UI["color_casos"])
    plt.tight_layout(); plt.savefig(OUTDIR / "figs" / "confusion_TOTAL.png"); plt.close()

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
    plt.tight_layout(); plt.savefig(OUTDIR / "figs" / "confusion_norm_TOTAL.png"); plt.close()

    # ---------- Curvas ROC / PR / Calibración (si hay proba) ----------
    if has_proba and y_proba is not None:
        # ROC OVR
        from sklearn.preprocessing import label_binarize
        Y_bin = label_binarize(y, classes=LEVELS)
        plt.figure(figsize=(6.2,4.6), dpi=160)
        for i, cls in enumerate(LEVELS):
            fpr, tpr, _ = roc_curve(Y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
        plt.plot([0,1],[0,1],"k--", linewidth=1)
        plt.xlabel(UI["roc_x"]); plt.ylabel(UI["roc_y"]); plt.title(UI["roc_title"])
        plt.legend(); plt.tight_layout(); plt.savefig(OUTDIR / "figs_ml" / "roc_ovr_TOTAL.png"); plt.close()

        # PR OVR
        plt.figure(figsize=(6.2,4.6), dpi=160)
        for i, cls in enumerate(LEVELS):
            precision, recall, _ = precision_recall_curve(Y_bin[:, i], y_proba[:, i])
            ap = average_precision_score(Y_bin[:, i], y_proba[:, i])
            plt.plot(recall, precision, label=f"{cls} (AP={ap:.2f})")
        plt.xlabel(UI["pr_x"]); plt.ylabel(UI["pr_y"]); plt.title(UI["pr_title"])
        plt.legend(); plt.tight_layout(); plt.savefig(OUTDIR / "figs_ml" / "pr_ovr_TOTAL.png"); plt.close()

        # Calibración 'Muy alto'
        try:
            idx_muy = LEVELS.index("Muy alto")
            prob_muy = y_proba[:, idx_muy]
            y_muy = (np.array(y) == "Muy alto").astype(int)
            frac_pos, mean_pred = calibration_curve(y_muy, prob_muy, n_bins=8, strategy="uniform")
            plt.figure(figsize=(6.2,4.6), dpi=160)
            plt.plot(mean_pred, frac_pos, marker="o", label=UI["cal_leg1"])
            plt.plot([0,1],[0,1],"k--", label=UI["cal_leg2"])
            plt.xlabel(UI["cal_x"]); plt.ylabel(UI["cal_y"]); plt.title(UI["cal_title"])
            plt.legend(); plt.tight_layout(); plt.savefig(OUTDIR / "figs_ml" / "calibracion_muy_alto_TOTAL.png"); plt.close()
        except Exception:
            pass

    # ---------- Estabilidad CV ----------
    cv_f1 = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")
    pd.DataFrame({"fold": list(range(1,len(cv_f1)+1)), "f1_macro": cv_f1})\
      .to_csv(OUTDIR / "tables_ml" / "cv_f1_folds_TOTAL.csv", index=False, encoding="utf-8")
    plt.figure(figsize=(4.8,4.6), dpi=160)
    plt.boxplot(cv_f1, vert=True, labels=["F1_macro"])
    plt.title(f"{UI['cv_title']}\nMedia={cv_f1.mean():.3f}, SD={cv_f1.std():.3f}")
    plt.ylabel(UI["cv_y"])
    plt.tight_layout(); plt.savefig(OUTDIR / "figs_ml" / "cv_boxplot_f1_TOTAL.png"); plt.close()

    # ---------- Curva de aprendizaje ----------
    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=skf, scoring="f1_macro",
        train_sizes=np.linspace(0.2, 1.0, 5), shuffle=True, random_state=RANDOM_STATE
    )
    plt.figure(figsize=(6.6,4.8), dpi=160)
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Entrenamiento")
    plt.plot(train_sizes, test_scores.mean(axis=1), marker="o", label="Validación (CV)")
    plt.xlabel(UI["learn_x"]); plt.ylabel(UI["learn_y"]); plt.legend()
    plt.title(UI["learn_title"])
    plt.tight_layout(); plt.savefig(OUTDIR / "figs_ml" / "learning_curve_TOTAL.png"); plt.close()

    # ---------- Importancias por permutación (sobre el pipeline entrenado) ----------
    # Ajustamos el pipeline con TODOS los datos para calcular importancias
    pipe.fit(X, y)
    r = permutation_importance(
        pipe, X, y,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring='f1_macro'
    )

    # Tabla completa de importancias
    importancias = pd.DataFrame({
        "feature": X.columns,
        "importance": r.importances_mean
    }).sort_values("importance", ascending=False)
    importancias.to_csv(OUTDIR / "figs_ml" / "importancias_items_total.csv",
                        index=False, encoding="utf-8")

    # (1) Top-N por ítem
    top_n = 15
    top = importancias.head(top_n).iloc[::-1]
    plt.figure(figsize=(7.0, 5.0), dpi=160)
    plt.barh(top["feature"], top["importance"])
    plt.title(UI["imp_items_title"])
    plt.xlabel(UI["imp_items_x"]); plt.ylabel(UI["imp_items_y"])
    plt.tight_layout(); plt.savefig(OUTDIR / "figs_ml" / "importancias_items_total.png"); plt.close()

    # (2) Agregado por subescala + edad/genero
    inv_map = {}
    for dim, idxs in DIMENSIONES.items():
        for i in idxs:
            inv_map[f"p{i}"] = dim

    def bucket(f):
        if f in ("edad", "genero"):
            return f
        return inv_map.get(f, "Otros")

    agg = (importancias
           .assign(grupo=importancias["feature"].map(bucket))
           .groupby("grupo", as_index=False)["importance"].sum()
           .sort_values("importance", ascending=False))

    plt.figure(figsize=(7.0, 4.6), dpi=160)
    plt.barh(agg["grupo"].iloc[::-1], agg["importance"].iloc[::-1])
    plt.title(UI["imp_dim_title"])
    plt.xlabel(UI["imp_dim_x"]); plt.ylabel(UI["imp_dim_y"])
    plt.tight_layout(); plt.savefig(OUTDIR / "figs_ml" / "importancias_dim_total.png"); plt.close()

    # ---------- Resumen JSON ----------
    resumen = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_registros_usados": int(len(y)),
        "levels_order": LEVELS,
        "metrics": {
            "f1_macro": float(f1m),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal),
            "auc_ovr_macro": None if auc_macro is None else float(auc_macro),
            "labels_order": LEVELS,
            "confusion_matrix": cm.tolist()
        }
    }
    with open(OUTDIR / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print("\n✅ Reportes (TOTAL) generados en:", OUTDIR.resolve())
    print(f"   F1={f1m:.3f}  Acc={acc:.3f}  BalAcc={bal:.3f}  "
          f"AUC={auc_macro:.3f}" if auc_macro is not None else
          f"   F1={f1m:.3f}  Acc={acc:.3f}  BalAcc={bal:.3f}  AUC=NA")

# ==================== MAIN ====================
def main():
    generar_reportes_total()

if __name__ == "__main__":
    main()
