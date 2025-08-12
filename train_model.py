# train_model.py
# Entrena un modelo con scas_respuestas.csv y guarda:
#   - model.pkl
#   - label_encoder.pkl
#   - feature_cols.pkl

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import joblib

CSV_PATH = "scas_respuestas.csv"

# 1) Cargar datos
df = pd.read_csv(CSV_PATH)

# 2) Definir features y etiqueta
feature_cols = [f"p{i}" for i in range(1, 39)]
assert all(col in df.columns for col in feature_cols), "Faltan columnas p1..p38 en el CSV."

X = df[feature_cols].fillna(0).astype(int)
y = df["nivel"].astype(str)

# 3) Codificar etiqueta (Normal/Elevado/Alto/Muy alto)
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 4) Split train/test (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 5) Modelo (robusto para empezar)
clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"  # por si las clases no quedan perfectamente balanceadas
)

clf.fit(X_train, y_train)

# 6) Evaluación rápida
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

print(f"Accuracy (holdout): {acc:.3f}")
print(f"F1-macro (holdout): {f1m:.3f}\n")

print("== Classification report ==")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("== Confusion matrix ==")
print(confusion_matrix(y_test, y_pred))

# 7) Guardar artefactos
joblib.dump(clf, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

print("\nArtefactos guardados: model.pkl, label_encoder.pkl, feature_cols.pkl")