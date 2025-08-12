# simulate_data.py
# Genera datos sintéticos SCAS coherentes con normas por género (12–15)
# Crea 40 registros por cada nivel: Normal, Elevado, Alto, Muy alto

import os
import random
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

def get_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )

# Ítems por dimensión (tu mapeo oficial)
DIMENSIONES = {
    "Dim1": [12, 19, 25, 27, 28, 30, 32, 33, 34],      # Pánico/Agorafobia (9)
    "Dim2": [5, 8, 11, 14, 15, 38],                    # Ansiedad por separación (6)
    "Dim3": [6, 7, 9, 10, 26, 31],                     # Fobia social (6)
    "Dim4": [2, 16, 21, 23, 29],                       # Miedo a lesiones físicas (5)
    "Dim5": [13, 17, 24, 35, 36, 37],                  # OCD (6)
    "Dim6": [1, 3, 4, 18, 20, 22],                     # Ansiedad generalizada (6)
}

# Cortes TOTAL por género (12–15)
GIRLS_TOTAL = {         # Niñas 12–15
    "normal": (0, 39),
    "elevado": (40, 52),
    "alto": (53, 66),
    "muy_alto_min": 67
}
BOYS_TOTAL = {          # Niños 12–15
    "normal": (0, 32),
    "elevado": (33, 41),
    "alto": (42, 59),
    "muy_alto_min": 60
}

def nivel_por_total_genero(total: int, genero: str) -> str:
    """Devuelve Normal/Elevado/Alto/Muy alto según género (12–15)."""
    cuts = GIRLS_TOTAL if genero == "Femenino" else BOYS_TOTAL
    if total >= cuts["muy_alto_min"]:
        return "Muy alto"
    a, b = cuts["alto"]
    if a <= total <= b:
        return "Alto"
    a, b = cuts["elevado"]
    if a <= total <= b:
        return "Elevado"
    a, b = cuts["normal"]
    if a <= total <= b:
        return "Normal"
    return "Normal"  # fallback

def generar_respuesta_con_sesgo(objetivo: str, genero: str, max_intentos=3000):
    """
    Genera respuestas p1..p38 intentando que el TOTAL caiga en el rango objetivo
    (Normal/Elevado/Alto/Muy alto) según el género.
    """
    # Pesos base por objetivo (más alto => más prob de 2/3)
    sesgos = {
        "Normal":   [70, 20, 8, 2],   # prob para valores 0,1,2,3
        "Elevado":  [35, 35, 20, 10],
        "Alto":     [15, 30, 35, 20],
        "Muy alto": [5, 15, 40, 40],
    }
    pesos = sesgos[objetivo]

    resp = {}
    total = 0
    for _ in range(max_intentos):
        resp = {f"p{i}": random.choices([0,1,2,3], weights=pesos, k=1)[0] for i in range(1, 39)}
        total = sum(resp.values())
        etiqueta = nivel_por_total_genero(total, genero)
        if etiqueta == objetivo:
            return resp, total
    return resp, total  # último intento (raro que no encaje)

def sumar_dimensiones(respuestas):
    return {dim: sum(respuestas[f"p{i}"] for i in items) for dim, items in DIMENSIONES.items()}

def insertar_muestra(cur, nombre, edad, genero, respuestas, sumas, total, nivel):
    columnas_p = ", ".join([f"p{i}" for i in range(1, 39)])
    placeholders_p = ", ".join(["%s"] * 38)
    sql = f"""
    INSERT INTO cuestionario
    (nombre, edad, genero, {columnas_p},
     puntaje_Dim1, puntaje_Dim2, puntaje_Dim3,
     puntaje_Dim4, puntaje_Dim5, puntaje_Dim6,
     puntaje_total, nivel)
    VALUES (%s, %s, %s, {placeholders_p}, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    valores = [
        nombre, edad, genero,
        *[respuestas[f"p{i}"] for i in range(1, 39)],
        sumas["Dim1"], sumas["Dim2"], sumas["Dim3"],
        sumas["Dim4"], sumas["Dim5"], sumas["Dim6"],
        total, nivel
    ]
    cur.execute(sql, valores)

def main(n_normal=10, n_elevado=10, n_alto=10, n_muy_alto=10):
    cn = get_db()
    cur = cn.cursor()

    objetivos = [("Normal", n_normal), ("Elevado", n_elevado), ("Alto", n_alto), ("Muy alto", n_muy_alto)]
    idx = 1
    for objetivo, n in objetivos:
        for _ in range(n):
            genero = random.choice(["Femenino", "Masculino"])
            edad = random.randint(12, 15)  # para que apliquen normas
            resp, total = generar_respuesta_con_sesgo(objetivo, genero)
            sumas = sumar_dimensiones(resp)
            nivel = nivel_por_total_genero(total, genero)
            nombre = f"Sim_{objetivo}_{genero}_{idx}"
            insertar_muestra(cur, nombre, edad, genero, resp, sumas, total, nivel)
            idx += 1

    cn.commit()
    cur.close(); cn.close()
    print(f"Insertados simulados: Normal={n_normal}, Elevado={n_elevado}, Alto={n_alto}, Muy alto={n_muy_alto} "
          f"(total={n_normal+n_elevado+n_alto+n_muy_alto})")

if __name__ == "__main__":
    # 40 por nivel como pediste
    main(n_normal=10, n_elevado=10, n_alto=10, n_muy_alto=10)