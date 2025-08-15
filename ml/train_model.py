# ml/train_model.py
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from Scas.configuracion import get_db  # tu conexión existente (mysql-connector)

# p1..p38
PREGUNTAS = [f"p{i}" for i in range(1, 39)]

# === Consulta: último cuestionario por alumno + nivel (etiqueta) ===
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

def main():
    print("[ML] Iniciando entrenamiento…")

    # --- Cargar datos desde BD ---
    print("[ML] Conectando a BD…")
    cn = get_db()
    cur = cn.cursor(dictionary=True)
    cur.execute(SQL_ULTIMO_X_ALUMNO)
    rows = cur.fetchall()
    cur.close(); cn.close()

    df = pd.DataFrame(rows)
    print(f"[ML] Registros leídos: {len(df)}")

    # Filtrar filas con etiqueta disponible
    df = df.dropna(subset=["nivel"]).copy()

    # Normalizar etiqueta a 4 clases exactas
    def norm_label(s: str) -> str:
        s = (s or "").strip().lower()
        if "muy" in s:
            return "Muy alto"
        if "alto" == s:
            return "Alto"
        if "elev" in s:
            return "Elevado"
        return "Normal"
    df["nivel_norm"] = df["nivel"].map(norm_label)

    print("[ML] Distribución de clases:")
    print(df["nivel_norm"].value_counts().sort_index())

    # --- Features: p1..p38 + edad + genero ---
    feature_cols_num = PREGUNTAS + ["edad"]         # numéricas
    feature_cols_cat = ["genero"]                   # categórica

    X = df[feature_cols_num + feature_cols_cat].copy()
    y = df["nivel_norm"].copy()

    # Preprocesador (one-hot para género)
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ]
    )

    # Modelo
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    clf = Pipeline(steps=[("preprocessor", pre), ("model", model)])

    # Split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenar
    clf.fit(Xtr, ytr)

    # Evaluar
    ypred = clf.predict(Xte)
    print("\n=== Reporte (test) ===")
    print(classification_report(yte, ypred))

    # Guardar
    outdir = Path(__file__).parent / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model_v1.joblib"
    joblib.dump(clf, model_path)
    print(f"✅ Modelo guardado en: {model_path.resolve()}")

if __name__ == "__main__":
    main()
