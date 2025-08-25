# ml/preprocesamiento.py — D1 completa (I1–I5) con figuras y reportes
# Ejecuta desde la raíz del proyecto:
#   (.venv) > python -m ml.preprocesamiento

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

from Scas.configuracion import get_db  # conexión MySQL

# --- Config ---
MODEL_VERSION = "v1"
PREGUNTAS = [f"p{i}" for i in range(1, 39)]
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
MODEL_PATH = Path(__file__).resolve().parent / "models" / f"model_{MODEL_VERSION}.joblib"

# --- Consulta: último cuestionario por estudiante (para auditoría D1) ---
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

# =========================
# Indicadores I1–I3 (auditoría del dataset)
# =========================
def compute_I1(df: pd.DataFrame, cols: list[str]) -> dict:
    """I1: % de datos completos (global y por columna)."""
    total_cells = df[cols].shape[0] * df[cols].shape[1]
    n_nulls = int(df[cols].isna().sum().sum())
    pct_global = (1 - n_nulls / max(1, total_cells)) * 100
    pct_by_col = (1 - df[cols].isna().mean()) * 100
    return {
        "pct_global": round(float(pct_global), 2),
        "pct_by_col": pct_by_col.round(2).to_dict()
    }

def compute_I2_out_of_range(df: pd.DataFrame, items: list[str], low=0, high=3) -> dict:
    """I2 (parte A): % de registros fuera de rango (0–3) por ítem."""
    bad = ~df[items].apply(lambda s: s.between(low, high))
    pct_bad_by_item = (bad.mean() * 100).round(2)
    return {"pct_bad_by_item": pct_bad_by_item.to_dict()}

def compute_I2_duplicates(df: pd.DataFrame, key_cols: list[str]) -> dict:
    """I2 (parte B): % de duplicados con respecto a columnas clave."""
    pct_dup = df.duplicated(subset=key_cols, keep="first").mean() * 100
    return {"pct_duplicates": round(float(pct_dup), 2)}

def compute_I3_balance(series: pd.Series) -> dict:
    """I3: distribución de clases de la variable objetivo + índice B (min/max)."""
    s = series.fillna("Sin etiqueta")
    dist = (s.value_counts(normalize=True) * 100).sort_index()
    B = float(dist.min() / dist.max()) if len(dist) and dist.max() > 0 else 0.0
    return {"dist_pct": dist.round(1).to_dict(), "indice_B": round(B, 3)}

# =========================
# I4–I5: Metadatos del pipeline (modelo entrenado)
# =========================
def _contains_standard_scaler(obj) -> bool:
    """Detecta si hay StandardScaler en una cadena/objeto (por si el num_tf es un Pipeline)."""
    try:
        from sklearn.preprocessing import StandardScaler  # noqa
        # chequeo directo
        if obj.__class__.__name__ == "StandardScaler":
            return True
    except Exception:
        pass
    # Si es un Pipeline u objeto compuesto, recorre sus steps
    try:
        steps = getattr(obj, "steps", None)
        if steps:
            for _, step in steps:
                if _contains_standard_scaler(step):
                    return True
    except Exception:
        pass
    return False

def pipeline_metadata(model_path: Path) -> dict:
    """
    I4/I5: extrae información del preprocesador dentro del modelo:
      - n_cat_codificadas
      - n_num_transformadas
      - n_cols_post_ohe (salida del preprocesador)
      - scale_coverage_pct (0% si no hay StandardScaler)
    """
    meta = {
        "n_cat_codificadas": 0,
        "n_num_transformadas": 0,
        "n_cols_post_ohe": None,
        "scale_coverage_pct": 0.0,
        "nota": ""
    }
    try:
        clf = load(model_path)
    except Exception as e:
        meta["nota"] = f"No se pudo cargar el modelo: {e}"
        return meta

    pre = clf.named_steps.get("preprocessor", None) if hasattr(clf, "named_steps") else None
    if pre is None:
        meta["nota"] = "Pipeline sin paso 'preprocessor'."
        return meta

    num_tf, cat_tf = None, None
    for name, tfm, cols in pre.transformers_:
        if name == "num":
            num_tf = tfm
        elif name == "cat":
            cat_tf = tfm

    # I4: categóricas codificadas
    if cat_tf is not None:
        try:
            cat_cols_in = pre.named_transformers_["cat"].feature_names_in_.tolist()
        except Exception:
            cat_cols_in = ["genero"]
        meta["n_cat_codificadas"] = len(cat_cols_in)

    # I4/I5: numéricas transformadas + detección de escalado
    if (num_tf is not None) and (num_tf != "passthrough"):
        try:
            num_cols_in = pre.named_transformers_["num"].feature_names_in_.tolist()
        except Exception:
            num_cols_in = []
        meta["n_num_transformadas"] = len(num_cols_in)
        # ¿Hay StandardScaler en la rama numérica?
        if _contains_standard_scaler(num_tf):
            # cobertura = 100% si todas las numéricas pasan por el scaler;
            # aquí simplificamos a 100% si se detecta scaler
            meta["scale_coverage_pct"] = 100.0
    else:
        meta["scale_coverage_pct"] = 0.0  # RF: no se escala (correcto)

    # Nº columnas de salida del preprocesador
    try:
        meta["n_cols_post_ohe"] = int(len(pre.get_feature_names_out()))
    except Exception:
        meta["n_cols_post_ohe"] = None

    return meta

# =========================
# Utilidades de graficado
# =========================
def plot_bar_dict(d: dict, title: str, xlabel: str, ylabel: str, path: Path, rotate=0, yfmt="{:.0f}"):
    """Barras con etiquetas encima (acepta valores 0)."""
    keys = list(d.keys())
    vals = [float(d[k]) for k in keys]
    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=rotate, ha="right")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)

    # Etiquetas de valor sobre cada barra
    top = max(vals) if vals else 0.0
    offset = top * 0.02 if top > 0 else 0.5
    for x, b, v in zip(range(len(keys)), bars, vals):
        y = b.get_height()
        plt.text(x, y + offset, yfmt.format(v), ha="center", va="bottom", fontsize=9)

    # Margen superior para que no se corte la etiqueta
    if top > 0:
        plt.ylim(0, top * 1.15)

    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def plot_nulls_heatmap(df: pd.DataFrame, cols: list[str], path: Path):
    """Heatmap horizontal sencillo de % de nulos por columna (I1)."""
    pct_nulls = (df[cols].isna().mean() * 100).values.reshape(1, -1)
    plt.figure(figsize=(12, 2))
    plt.imshow(pct_nulls, aspect="auto")
    plt.colorbar(label="% nulos")
    plt.yticks([0], [""])
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.title("Porcentaje de nulos por columna (D1·I1)")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

# =========================
# Main
# =========================
def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Dataset base para auditoría (último por estudiante)
    cn = get_db()
    cur = cn.cursor(dictionary=True)
    cur.execute(SQL_ULTIMO_X_ALUMNO)
    rows = cur.fetchall()
    cur.close(); cn.close()

    df = pd.DataFrame(rows)
    print(f"[D1] Registros: {len(df)}  |  Columnas: {len(df.columns)}")

    item_cols = PREGUNTAS
    base_cols = item_cols + ["edad", "genero"]

    # 2) I1–I3 (auditoría de datos)
    I1 = compute_I1(df, base_cols)
    I2a = compute_I2_out_of_range(df, item_cols, 0, 3)
    I2b = compute_I2_duplicates(df, key_cols=base_cols)
    I3 = compute_I3_balance(df["nivel"])

    # 3) I4–I5 (metadatos del pipeline entrenado)
    I45 = pipeline_metadata(MODEL_PATH)

    # 4) Guardados numéricos (CSV + JSON)
    pd.Series(I1["pct_by_col"]).to_csv(REPORTS_DIR / f"d1_I1_pct_completos_por_col_{ts}.csv", encoding="utf-8-sig")
    pd.Series(I2a["pct_bad_by_item"]).to_csv(REPORTS_DIR / f"d1_I2_fuera_rango_por_item_{ts}.csv", encoding="utf-8-sig")
    pd.Series(I3["dist_pct"]).to_csv(REPORTS_DIR / f"d1_I3_distribucion_clases_{ts}.csv", encoding="utf-8-sig")

    resumen = {
        "model_version": MODEL_VERSION,
        "timestamp": ts,
        "n_registros": int(len(df)),
        "I1": I1,
        "I2": {"out_of_range": I2a, "duplicates": I2b},
        "I3": I3,
        "I4_I5": I45,
        "n_features_antes": len(PREGUNTAS) + 2,  # p1..p38 + edad + genero
        "n_features_despues": I45["n_cols_post_ohe"],
    }
    (REPORTS_DIR / f"d1_resumen_{ts}.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5) Figuras PNG (con etiquetas numéricas)
    # I1
    plot_nulls_heatmap(df, base_cols, REPORTS_DIR / f"d1_I1_heatmap_nulos_{ts}.png")

    # I2 (porcentaje con 2 decimales)
    plot_bar_dict(I2a["pct_bad_by_item"],
                  "Fuera de rango por ítem (D1·I2)", "Ítems SCAS", "% registros",
                  REPORTS_DIR / f"d1_I2_fuera_rango_{ts}.png",
                  rotate=90, yfmt="{:.2f}%")

    # I3 (porcentaje con 1 decimal)
    plot_bar_dict(I3["dist_pct"],
                  "Distribución de clases (D1·I3)", "Clase", "%",
                  REPORTS_DIR / f"d1_I3_distribucion_{ts}.png",
                  yfmt="{:.1f}%")

    # I4
    plot_bar_dict(
        {"cat_codificadas": I45["n_cat_codificadas"], "num_transformadas": I45["n_num_transformadas"]},
        "Variables codificadas/transformadas (D1·I4)", "Tipo", "Cantidad",
        REPORTS_DIR / f"d1_I4_codif_transf_{ts}.png",
        yfmt="{:.0f}"
    )
    plot_bar_dict(
        {"antes": len(PREGUNTAS) + 2, "despues": I45["n_cols_post_ohe"] or 0},
        "Características antes vs. después del preprocesamiento (D1·I4)",
        "Estado", "N° de características",
        REPORTS_DIR / f"d1_I4_antes_despues_{ts}.png",
        yfmt="{:.0f}"
    )

    # I5 (porcentaje con 1 decimal)
    plot_bar_dict(
        {"cobertura_escalado_%": I45["scale_coverage_pct"]},
        "Cobertura de escalado (D1·I5)", "", "%",
        REPORTS_DIR / f"d1_I5_escalado_{ts}.png",
        yfmt="{:.1f}%"
    )

    # 6) Deja también el dataset base para trazabilidad (útil en anexos)
    df.to_csv(REPORTS_DIR / f"d1_dataset_base_{ts}.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ D1 completo. Revisa PNG/CSV/JSON en: {REPORTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
