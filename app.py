#Framework web para mostrar el formulario y manejar las respuestas.
from flask import Flask, render_template, request, redirect
#Manipulación de datos 
import pandas as pd
import os

# importa tu conexión BD y reglas desde el paquete scas
from Scas.configuracion import get_db
from Scas.regla_puntuaciones import (
    DIMENSIONES, DIM_NOMBRES, PRETTY,
    interpreta_normas,
)

#----------------------------------------------
#Inicializar la app Flask
app = Flask(__name__)
#----------------------------------------------
# === 8) Rutas ===
@app.route('/')
def home():
    return render_template("index.html")

# Ruta para mostrar el formulario registro
@app.route('/form_registro')
def form_registro():
    return render_template("registro.html")

# Ruta para login
@app.route('/form_login')
def form_login():
    return render_template("login.html")

# /cuestionario ahora exige ?uid=... y lo pasa al template
@app.route('/cuestionario')
def cuestionario():
    uid = request.args.get('uid', type=int)
    if not uid:
        # si no hay uid en la URL, mejor redirige al login
        return redirect('/form_login')
    return render_template("cuestionario.html", uid=uid)

# Ruta para Resultado
@app.route('/resultado')
def resultado():
    uid = request.args.get('uid', type=int)
    if not uid:
        return "Falta uid", 400

    cn = get_db()
    cur = cn.cursor(dictionary=True)
    cur.execute("""
        SELECT 
            u.id_usuario, u.nombre,
            c.puntaje_Dim1, c.puntaje_Dim2, c.puntaje_Dim3,
            c.puntaje_Dim4, c.puntaje_Dim5, c.puntaje_Dim6,
            c.puntaje_total, c.nivel, c.created_at
        FROM usuario u
        LEFT JOIN cuestionario c ON c.id_usuario = u.id_usuario
        WHERE u.id_usuario = %s
        ORDER BY c.created_at DESC
        LIMIT 1
    """, (uid,))
    row = cur.fetchone()
    cur.close(); cn.close()

    if not row:
        # Usuario sin registros
        return render_template("resultado.html", uid=uid, nombre=None, no_data=True)

    if row.get("puntaje_total") is None:
        # Tiene usuario pero sin cuestionario aún
        return render_template("resultado.html", uid=row["id_usuario"], nombre=row["nombre"], no_data=True)

    subdim = {
        "Pánico/Agorafobia":         row["puntaje_Dim1"],
        "Ansiedad por separación":   row["puntaje_Dim2"],
        "Fobia social":              row["puntaje_Dim3"],
        "Miedo a lesiones físicas":  row["puntaje_Dim4"],
        "Obs.-Compulsivo (OCD)":     row["puntaje_Dim5"],
        "Ansiedad generalizada":     row["puntaje_Dim6"],
    }

    return render_template(
        "resultado.html",
        uid=row["id_usuario"],              # <- ID disponible para la vista
        nombre=row["nombre"],               # <- nombre visible
        subdim=subdim,
        total=row["puntaje_total"],
        nivel=row["nivel"],
        creado=row["created_at"],
        no_data=False,
    )

# Ruta para que guarde el registro de usuario (GET y POST)
@app.route('/registro', methods=['GET', 'POST'])
def registro():
    if request.method == 'GET':
        # muestra la vista normal
        return render_template("registro.html", exito=False, error=None)

    # POST: datos desde el formulario
    nombre   = request.form.get("nombre")
    email    = request.form.get("email")
    password = request.form.get("password")  # (sin hash, como pediste)

    cn = get_db()
    cur = cn.cursor()

    try:
        cur.execute(
            "INSERT INTO usuario (nombre, email, contraseña) VALUES (%s, %s, %s)",
            (nombre, email, password)
        )
        cn.commit()
        # <- volvemos a la MISMA vista con la bandera exito=True
        return render_template("registro.html", exito=True, error=None)

    except Exception as e:
        cn.rollback()
        # podrías mapear errores (p.ej. email duplicado) a un mensaje más lindo
        return render_template("registro.html", exito=False, error=str(e))

    finally:
        cur.close()
        cn.close()

# === Login (GET/POST) ===
# IMPORTANTE: tu login.html debe postear a /login (action="/login")
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html', error=None)

    # POST: validar contra BD (sin hash)
    email = request.form.get('email')
    password = request.form.get('password')

    cn = get_db()
    cur = cn.cursor()
    try:
        cur.execute(
            "SELECT id_usuario FROM usuario WHERE email=%s AND contraseña=%s",
            (email, password)
        )
        row = cur.fetchone()
    finally:
        cur.close()
        cn.close()

    if row:
        uid = row[0]
        # pasa el id a /cuestionario vía query string
        return redirect(f'/cuestionario?uid={uid}')

    # credenciales incorrectas
    return render_template('login.html', error="Correo o contraseña incorrectos.")

# === Guardar/Actualizar cuestionario ===
@app.post('/guardar')
def guardar():
    try:
        # Validar que venga id_usuario (oculto en el form)
        id_usuario_raw = (request.form.get("id_usuario") or "").strip()
        if not id_usuario_raw.isdigit():
            return "Falta id_usuario. Vuelve a iniciar sesión.", 400
        id_usuario = int(id_usuario_raw)

        # Datos demográficos
        edad = int(request.form.get("edad"))
        genero = request.form.get("genero")  # "Femenino" / "Masculino"

        # Respuestas p1..p38 (cada una 0..3)
        respuestas = {f"p{i}": int(request.form.get(f"p{i}", 0)) for i in range(1, 39)}

        # Sumas por dimensión y total
        sumas = {dim: sum(respuestas[f"p{i}"] for i in items) for dim, items in DIMENSIONES.items()}
        puntaje_total = sum(respuestas.values())

        # Interpretación por reglas
        inter_sub, inter_total = interpreta_normas(genero, edad, sumas, puntaje_total)
        nivel_txt = inter_total or "N/A"

        cn = get_db()
        cur = cn.cursor()

        # ¿Ya tiene cuestionario? Tomar el más reciente si existiera
        cur.execute(
            "SELECT id_cuestionario FROM cuestionario WHERE id_usuario=%s ORDER BY created_at DESC LIMIT 1",
            (id_usuario,)
        )
        row = cur.fetchone()

        if row:
            # UPDATE sobre el existente
            id_cuest = row[0]
            set_cols_p = ", ".join([f"p{i}=%s" for i in range(1, 39)])
            sql = f"""
                UPDATE cuestionario
                SET edad=%s, genero=%s, {set_cols_p},
                    puntaje_Dim1=%s, puntaje_Dim2=%s, puntaje_Dim3=%s,
                    puntaje_Dim4=%s, puntaje_Dim5=%s, puntaje_Dim6=%s,
                    puntaje_total=%s, nivel=%s
                WHERE id_cuestionario=%s
            """
            valores = [
                edad, genero,
                *[respuestas[f"p{i}"] for i in range(1, 39)],
                sumas["Dim1"], sumas["Dim2"], sumas["Dim3"],
                sumas["Dim4"], sumas["Dim5"], sumas["Dim6"],
                puntaje_total, nivel_txt,
                id_cuest
            ]
            cur.execute(sql, valores)
        else:
            # INSERT nuevo
            columnas_p = ", ".join([f"p{i}" for i in range(1, 39)])
            placeholders_p = ", ".join(["%s"] * 38)
            sql = f"""
                INSERT INTO cuestionario
                (id_usuario, edad, genero, {columnas_p},
                 puntaje_Dim1, puntaje_Dim2, puntaje_Dim3,
                 puntaje_Dim4, puntaje_Dim5, puntaje_Dim6,
                 puntaje_total, nivel)
                VALUES (%s, %s, %s, {placeholders_p}, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            valores = [
                id_usuario, edad, genero,
                *[respuestas[f"p{i}"] for i in range(1, 39)],
                sumas["Dim1"], sumas["Dim2"], sumas["Dim3"],
                sumas["Dim4"], sumas["Dim5"], sumas["Dim6"],
                puntaje_total, nivel_txt
            ]
            cur.execute(sql, valores)

        cn.commit()
        cur.close(); cn.close()

        # volver al mismo cuestionario del usuario (puedes cambiarlo a una página de resultados)
        return redirect(f"/resultado?uid={id_usuario}")

    except Exception as e:
        return f"Error al guardar: {e}", 400

# === 9) Run ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))   # Render define PORT; 5000 de fallback local
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
