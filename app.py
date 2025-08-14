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

# Ruta para el cuestionario
@app.route('/cuestionario')
def cuestionario():
    return render_template("cuestionario.html")

# Ruta para registro (GET y POST)
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

# --- Login (GET+POST): funciona en /login y /form_login ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html', error=None)

    # POST: validar contra la BD (sin hash, como pediste)
    email = request.form.get('email')
    password = request.form.get('password')

    cn = get_db()
    cur = cn.cursor()
    cur.execute(
        "SELECT id_usuario FROM usuario WHERE email=%s AND contraseña=%s",
        (email, password)
    )
    row = cur.fetchone()
    cur.close(); cn.close()
    if row:
        # credenciales correctas -> ir al cuestionario
        return redirect('/cuestionario')

    # incorrectas -> volver al login con mensaje
    return render_template('login.html', error="Correo o contraseña incorrectos.")

# === 9) Run ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))   # Render define PORT; 5000 de fallback local
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
