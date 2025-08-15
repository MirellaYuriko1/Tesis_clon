# ml/train_model.py
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from Scas.configuracion import get_db

# columnas p1..p38
PREGUNTAS = [f"p{i}" for i in range(1, 39)]

SQL_ULTIMO_X_ALUMNO = f"""
    SELECT c.id_usuario, c.id_cuestionario, c.edad, c.genero,
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
    LEFT JOIN resultado r ON r.id_cuestionario = c.id_cuestionario
"""

def _normaliza_nivel(lbl):
    if not isinstance(lbl, str): return None
    return {
        "normal": "Normal",
        "elevado": "Elevado",
        "alto": "Alto",
        "muy alto": "Muy alto",
    }.get(lbl.strip().lower())

def fetch_data():
    print("[ML] Conectando a BD…")
    cn = get_db()
    cur = cn.cursor(dictionary=True)
    cur.execute("SELECT DATABASE() AS db, @@hostname AS host")
    info = cur.fetchone()
    print(f"[ML] Conectado a: {info['db']} @ {info['host']}")
    cur.execute(SQL_ULTIMO_X_ALUMNO)
    rows = cur.fetchall()
    cur.close(); cn.close()
    df = pd.DataFrame(rows)
    print(f"[ML] Registros leídos: {len(df)}")
    return df

def build_dataset(df: pd.DataFrame):
    df = df.copy()
    df["nivel_norm"] = df["nivel"].apply(_normaliza_nivel)
    df = df[df["nivel_norm"].notna()].reset_index(drop=True)
    if df.empty:
        raise SystemExit("❌ No hay filas con 'nivel' válido en 'resultado'.")

    # asegurar numérico en p1..p38
    for p in PREGUNTAS:
        df[p] = pd.to_numeric(df[p], errors="coerce").fillna(0).astype(float)

    X = df[PREGUNTAS]
    y = df["nivel_norm"]

    print("[ML] Distribución de clases:")
    print(y.value_counts())
    if y.nunique() < 2:
        raise SystemExit("❌ Se requieren ≥2 clases distintas para entrenar.")
    return X, y

def main():
    print("[ML] Iniciando entrenamiento…")
    df = fetch_data()
    if df.empty:
        print("❌ No hay datos.")
        return

    X, y = build_dataset(df)

    # si alguna clase tiene muy pocas muestras en test, usa 25%
    min_class = y.value_counts().min()
    test_size = 0.25 if min_class < 3 else 0.20

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=42
    )
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)

    print("\n=== Reporte (test) ===")
    print(classification_report(yte, ypred))

    outdir = Path(__file__).parent / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model_v1.joblib"
    joblib.dump(clf, model_path)
    print(f"✅ Modelo guardado en: {model_path.resolve()}")

if __name__ == "__main__":
    main()
