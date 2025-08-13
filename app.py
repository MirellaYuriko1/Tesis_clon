# --- Librerías y conexión MySQL ---
import os
import mysql.connector
from dotenv import load_dotenv
#------------------------------

#Framework web para mostrar el formulario y manejar las respuestas.
from flask import Flask, render_template, request 
#Manipulación de datos 
import pandas as pd
#ML
import joblib

#----------------------------------------------
# 1) Cargar variables del .env
load_dotenv() #lee tu archivo .env y carga esas variables en la memoria del sistema.

# 2) Función de conexión a MySQL
def get_db():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "3306")),  # <--- importante
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        # ssl_disabled=True  # si tu endpoint exige sin SSL. Si falla, pruébalo.
    )

#----------------------------------------------
#Inicializar la app Flask
app = Flask(__name__)
#----------------------------------------------

# Cargar artefactos del modelo (al iniciar la app)
MODEL = joblib.load("model.pkl")
LABELER = joblib.load("label_encoder.pkl")
FEATURE_COLS = joblib.load("feature_cols.pkl")

# === 4) Mapeo oficial de tus dimensiones (ítems SCAS) ===
# Dim1 = Pánico/Agorafobia
# Dim2 = Ansiedad por separación
# Dim3 = Fobia social
# Dim4 = Miedo a lesiones físicas
# Dim5 = Obsesivo-compulsivo (OCD)
# Dim6 = Ansiedad generalizada
DIMENSIONES = {
    "Dim1": [12, 19, 25, 27, 28, 30, 32, 33, 34],      # 9 ítems
    "Dim2": [5, 8, 11, 14, 15, 38],                    # 6 ítems
    "Dim3": [6, 7, 9, 10, 26, 31],                     # 6 ítems
    "Dim4": [2, 16, 21, 23, 29],                       # 5 ítems
    "Dim5": [13, 17, 24, 35, 36, 37],                  # 6 ítems
    "Dim6": [1, 3, 4, 18, 20, 22],                     # 6 ítems
}

# Nombres bonitos
DIM_NOMBRES = {
    "Dim1": "PanicoAgorafobia",
    "Dim2": "AnsiedadSeparacion",
    "Dim3": "FobiaSocial",
    "Dim4": "MiedoLesiones",
    "Dim5": "OCD",
    "Dim6": "AnsiedadGeneral",
}
PRETTY = {
    "PanicoAgorafobia": "Pánico/Agorafobia",
    "AnsiedadSeparacion": "Ansiedad por separación",
    "FobiaSocial": "Fobia social",
    "MiedoLesiones": "Miedo a lesiones físicas",
    "OCD": "Obs.-Compulsivo (OCD)",
    "AnsiedadGeneral": "Ansiedad generalizada",
}
# =============================================

# === 5) Predicción ML (nivel) ===
def predecir_nivel_ml(respuestas_dict):
    """
    respuestas_dict: {'p1':0, ..., 'p38':3}
    Devuelve: (nivel_txt, prob_max)
    """
    x = [[int(respuestas_dict[col]) for col in FEATURE_COLS]]
    y_pred_enc = MODEL.predict(x)[0]
    proba = MODEL.predict_proba(x)[0].max() if hasattr(MODEL, "predict_proba") else None
    nivel_txt = LABELER.inverse_transform([y_pred_enc])[0]
    return nivel_txt, proba

# === 6) Cortes SCAS oficiales: NIÑAS 12–15 (tu corrección) ===
GIRLS_12_15_CUTS = {
    "OCD":               {"normal": (0, 6), "elevado": (7, 9),  "alto": (10, 12), "muy_alto_min": 13},
    "FobiaSocial":       {"normal": (0, 9), "elevado": (10, 11),"alto": (12, 14), "muy_alto_min": 15},
    "PanicoAgorafobia":  {"normal": (0, 6), "elevado": (7, 9),  "alto": (10, 13), "muy_alto_min": 14},
    "AnsiedadSeparacion":{"normal": (0, 5), "elevado": (6, 7),  "alto": (8, 9),   "muy_alto_min": 10},
    "MiedoLesiones":     {"normal": (0, 5), "elevado": (6, 7),  "alto": (8, 9),   "muy_alto_min": 10},
    "AnsiedadGeneral":   {"normal": (0, 8), "elevado": (9, 10), "alto": (11, 14), "muy_alto_min": 15},
}
GIRLS_12_15_TOTAL = {
    "normal": (0, 39),
    "elevado": (40, 52),
    "alto": (53, 66),
    "muy_alto_min": 67
}

# === 7) Cortes SCAS oficiales: NIÑOS 12–15 (tu corrección) ===
BOYS_12_15_CUTS = {
    "OCD":               {"normal": (0, 6), "elevado": (7, 8),  "alto": (9, 11),  "muy_alto_min": 12},
    "FobiaSocial":       {"normal": (0, 7), "elevado": (8, 10), "alto": (11, 12), "muy_alto_min": 13},
    "PanicoAgorafobia":  {"normal": (0, 3), "elevado": (4, 7),  "alto": (8, 12),  "muy_alto_min": 13},
    "AnsiedadSeparacion":{"normal": (0, 3), "elevado": (4, 5),  "alto": (6, 8),   "muy_alto_min": 9},
    "MiedoLesiones":     {"normal": (0, 3), "elevado": (4, 5),  "alto": (6, 8),   "muy_alto_min": 9},
    "AnsiedadGeneral":   {"normal": (0, 6), "elevado": (7, 9),  "alto": (10, 12), "muy_alto_min": 13},
}
BOYS_12_15_TOTAL = {
    "normal": (0, 32),
    "elevado": (33, 41),
    "alto": (42, 59),
    "muy_alto_min": 60
}

def _clasifica_por_cortes(valor: int, cortes: dict) -> str:
    """Devuelve Normal / Elevado / Alto / Muy alto usando rangos inclusivos."""
    if valor >= cortes["muy_alto_min"]:
        return "Muy alto"
    a, b = cortes["alto"]
    if a <= valor <= b:
        return "Alto"
    a, b = cortes["elevado"]
    if a <= valor <= b:
        return "Elevado"
    a, b = cortes["normal"]
    if a <= valor <= b:
        return "Normal"
    return "Intermedio"

def interpreta_normas(genero: str, edad: int, sumas_dim: dict, total: int):
    """
    Retorna (dict_subescalas, texto_total) según género/edad.
    Solo aplica normas si (12 <= edad <= 15) y genero es Masculino/Femenino.
    """
    if 12 <= edad <= 15:
        if genero == "Femenino":
            cuts_dim = GIRLS_12_15_CUTS
            cuts_total = GIRLS_12_15_TOTAL
        elif genero == "Masculino":
            cuts_dim = BOYS_12_15_CUTS
            cuts_total = BOYS_12_15_TOTAL
        else:
            return {}, None

        inter_sub = {}
        for dim in ["Dim1","Dim2","Dim3","Dim4","Dim5","Dim6"]:
            nombre_sub = DIM_NOMBRES[dim]
            inter_sub[nombre_sub] = _clasifica_por_cortes(sumas_dim[dim], cuts_dim[nombre_sub])

        inter_total = _clasifica_por_cortes(total, cuts_total)
        return inter_sub, inter_total

    return {}, None

# === 8) Rutas ===
@app.route('/')
def home():
    return render_template("index.html")

# Ruta para mostrar el formulario registro
@app.route('/form_registro')
def form_registro():
    return render_template("registro.html")

# Ruta para login
@app.route('/login')
def login():
    return render_template("login.html")


# Ruta para el cuestionario
@app.route('/cuestionario')
def cuestionario():
    return render_template("cuestionario.html")

# Ruta para registro (GET y POST)
@app.route('/registro', methods=['GET', 'POST'])
def registro():
    if request.method == 'GET':
        return render_template("registro.html")

    # POST: datos desde el formulario
    nombre   = request.form.get("nombre")
    email    = request.form.get("email")
    password = request.form.get("password")  # Sin hash, se guarda directo

    cn = get_db()
    cur = cn.cursor()

    try:
        cur.execute(
            "INSERT INTO usuario (nombre, email, contraseña) VALUES (%s, %s, %s)",
            (nombre, email, password)
        )
        cn.commit()
        return "Registro exitoso. <a href='/login'>Inicia sesión aquí</a>."
    except Exception as e:
        cn.rollback()
        return f"Error al registrar: {e}"
    finally:
        cur.close()
        cn.close()

# === AL DARLE CLICK AL BOTON GUARDAR EL CUESTIONARIO
@app.route('/guardar', methods=['POST'])
def guardar():
    nombre = request.form.get("nombre")
    edad = int(request.form.get("edad"))
    genero = request.form.get("genero")

    # Respuestas p1..p38
    respuestas = {f"p{i}": int(request.form.get(f"p{i}")) for i in range(1, 39)}

    # Sumas por dimensión y total SCAS
    sumas = {dim: sum(respuestas[f"p{i}"] for i in items) for dim, items in DIMENSIONES.items()}
    puntaje_total = sum(respuestas.values())

    # Predicción ML
    nivel_ml, proba = predecir_nivel_ml(respuestas)

    # Interpretación normativa según género/edad
    interpretacion_sub, interpretacion_total = interpreta_normas(genero, edad, sumas, puntaje_total)

    # Guardar (nivel = ML)
    cn = get_db()
    cur = cn.cursor()

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
        puntaje_total, nivel_ml
    ]

    cur.execute(sql, valores)
    cn.commit()
    cur.close(); cn.close()

    # HTML de salida
    html_sub = ""
    if interpretacion_sub:
        filas = []
        for dim in ["Dim1","Dim2","Dim3","Dim4","Dim5","Dim6"]:
            nombre_sub = DIM_NOMBRES[dim]
            filas.append(
                f"<tr><td>{PRETTY[nombre_sub]}</td><td>{sumas[dim]}</td><td>{interpretacion_sub[nombre_sub]}</td></tr>"
            )
        html_sub = f"""
        <h3>Interpretación normativa ({'Niña' if genero=='Femenino' else 'Niño'} 12–15)</h3>
        <table border="1" cellpadding="6" cellspacing="0">
            <tr><th>Subescala</th><th>Puntaje</th><th>Nivel</th></tr>
            {''.join(filas)}
        </table>
        <p><b>Total SCAS:</b> {puntaje_total} → <b>{interpretacion_total}</b></p>
        """

    return f"""
    <h2>Gracias, {nombre}.</h2>
    <p>Edad: {edad} | Género: {genero}</p>

    <h3>Predicción (Machine Learning)</h3>
    <p>Nivel (ML): <b>{nivel_ml}</b> {f"(confianza: {proba:.2f})" if proba is not None else ""}</p>

    {html_sub if html_sub else "<p>(Sin interpretación normativa: solo aplica 12–15 años con género definido)</p>"}

    <p>¡Registro guardado en MySQL!</p>
    """

@app.route('/export_csv')
def export_csv():
    try:
        cn = get_db()
        query = """
            SELECT id, nombre, edad, genero,
                   p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,
                   p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,p37,p38,
                   puntaje_Dim1, puntaje_Dim2, puntaje_Dim3, puntaje_Dim4, puntaje_Dim5, puntaje_Dim6,
                   puntaje_total, nivel, created_at
            FROM cuestionario
            ORDER BY id ASC;
        """
        df = pd.read_sql(query, cn)
        cn.close()
        out_name = "scas_respuestas.csv"
        df.to_csv(out_name, index=False, encoding="utf-8")
        return f"Exportado correctamente a <b>{out_name}</b> con {len(df)} registros."
    except Exception as e:
        return f"Error al exportar: {e}"

# === 9) Run ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))   # Render define PORT; 5000 de fallback local
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
