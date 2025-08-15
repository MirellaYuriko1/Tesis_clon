#Framework web para mostrar el formulario y manejar las respuestas.
from flask import Flask, render_template, request, redirect, Response
from io import BytesIO

#==================================================
#==========INTEGRACION MACHINE LEARNING
#Manipulación de datos 
import pandas as pd
import os

# --- ML: cargar modelo y utilidades ---
from joblib import load
from pathlib import Path

PREGUNTAS = [f"p{i}" for i in range(1, 39)]
MODEL_PATH = Path(__file__).parent / "ml" / "models" / "model_v1.joblib"
_model = None

def get_model():
    """Carga el modelo una sola vez (lazy)."""
    global _model
    if _model is None:
        try:
            _model = load(MODEL_PATH)
            print(f"[ML] Modelo cargado: {MODEL_PATH}")
        except Exception as e:
            print(f"[ML] No se pudo cargar el modelo: {e}")
            _model = None
    return _model

def ml_predict_from_answers(respuestas: dict):
    """
    respuestas: dict con claves 'p1'..'p38' (int 0..3)
    Retorna (pred_label, proba_dict | None)
    """
    clf = get_model()
    if clf is None:
        return None, None

    X = [[float(respuestas.get(f"p{i}", 0)) for i in range(1, 39)]]
    pred = clf.predict(X)[0]

    proba = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        clases = list(clf.classes_)
        proba = {c: round(float(p) * 100, 1) for c, p in zip(clases, probs)}
    return pred, proba
#========================================================

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

# Ruta para mostrar el formulario cuestionario
@app.route('/cuestionario')
def cuestionario():
    uid = request.args.get('uid', type=int)
    if not uid:
        return redirect('/form_login')
    # Traer el nombre del usuario para mostrarlo en el navbar
    cn = get_db()
    cur = cn.cursor()
    cur.execute("SELECT nombre FROM usuario WHERE id_usuario=%s", (uid,))
    row = cur.fetchone()
    cur.close(); cn.close()
    usuario_nombre = row[0] if row else None
    return render_template('cuestionario.html', uid=uid, usuario_nombre=usuario_nombre)

# Ruta para ir al panel
@app.route('/form_panel')
def form_panel():
    uid = request.args.get('uid', type=int)
    q = (request.args.get('q') or '').strip()

    if not uid:
        return redirect('/form_login')

    cn = get_db()
    cur = cn.cursor(dictionary=True)
    try:
        # 1) Info del admin
        cur.execute("SELECT nombre, rol FROM usuario WHERE id_usuario=%s", (uid,))
        admin = cur.fetchone()
        if not admin:
            return "Usuario no encontrado.", 404

        if (admin.get('rol') or '').lower() != 'admin':
            # Si no es admin, enviar a su cuestionario
            return redirect(f'/cuestionario?uid={uid}')

        # 2) Último cuestionario por estudiante + su resultado (LEFT JOIN)
        where_like = ""
        params = []
        if q:
            where_like = " AND u.nombre LIKE %s "
            params.append(f"%{q}%")

        sql = f"""
            SELECT 
                u.id_usuario,
                u.nombre,
                c.genero,
                c.edad,
                r.puntaje_total,
                r.nivel,
                COALESCE(r.created_at, c.created_at) AS created_at
            FROM usuario u
            JOIN (
                SELECT c1.*
                FROM cuestionario c1
                JOIN (
                    SELECT id_usuario, MAX(created_at) AS mx
                    FROM cuestionario
                    GROUP BY id_usuario
                ) ult
                  ON ult.id_usuario = c1.id_usuario AND ult.mx = c1.created_at
            ) c ON c.id_usuario = u.id_usuario
            LEFT JOIN resultado r ON r.id_cuestionario = c.id_cuestionario
            WHERE u.rol = 'estudiante' {where_like}
            ORDER BY COALESCE(r.created_at, c.created_at) DESC
        """
        cur.execute(sql, params)
        rows = cur.fetchall()
    finally:
        cur.close()
        cn.close()

    return render_template(
        'panel.html',
        admin_nombre=admin['nombre'],
        rows=rows,
        uid=uid,
        q=q
    )

@app.get('/descargar_documento')
def descargar_documento():
    uid = request.args.get('uid', type=int)
    q = (request.args.get('q') or '').strip()
    if not uid:
        return redirect('/form_login')

    cn = get_db()
    cur = cn.cursor(dictionary=True)
    try:
        # validar admin
        cur.execute("SELECT rol FROM usuario WHERE id_usuario=%s", (uid,))
        rol_row = cur.fetchone()
        if not rol_row or (rol_row['rol'] or '').lower() != 'admin':
            return redirect(f'/cuestionario?uid={uid}')

        # armar consulta con TODAS las columnas de cuestionario (último por estudiante)
        cols_p = ", ".join([f"c.p{i}" for i in range(1, 39)])
        sql = f"""
            SELECT 
                u.nombre AS Nombre,
                c.genero AS Genero,
                c.edad   AS Edad,
                {cols_p},
                r.puntaje_Dim1, r.puntaje_Dim2, r.puntaje_Dim3,
                r.puntaje_Dim4, r.puntaje_Dim5, r.puntaje_Dim6,
                r.puntaje_total, r.nivel,
                COALESCE(r.created_at, c.created_at) AS created_at
            FROM usuario u
            JOIN (
                SELECT c1.*
                FROM cuestionario c1
                JOIN (
                    SELECT id_usuario, MAX(created_at) AS mx
                    FROM cuestionario
                    GROUP BY id_usuario
                ) ult
                  ON ult.id_usuario = c1.id_usuario AND ult.mx = c1.created_at
            ) c ON c.id_usuario = u.id_usuario
            LEFT JOIN resultado r ON r.id_cuestionario = c.id_cuestionario
            WHERE u.rol='estudiante' { "AND u.nombre LIKE %s" if q else "" }
            ORDER BY COALESCE(r.created_at, c.created_at) DESC
        """
        params = [f"%{q}%"] if q else []
        cur.execute(sql, params)
        data = cur.fetchall()
    finally:
        cur.close()
        cn.close()

    # Construir DataFrame y ordenar columnas
    if not data:
        df = pd.DataFrame(columns=["Nombre","Genero","Edad"])
    else:
        df = pd.DataFrame(data)

        # Orden elegante de columnas
        preguntas = [f"p{i}" for i in range(1, 39)]
        dims = [f"puntaje_Dim{i}" for i in range(1, 6+1)]
        front = ["Nombre", "Genero", "Edad"]
        tail = dims + ["puntaje_total", "nivel", "created_at"]
        ordered = [c for c in front + preguntas + tail if c in df.columns]
        df = df[ordered]

    # Exportar Excel en memoria
    try:
        buff = BytesIO()
        with pd.ExcelWriter(buff, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="SCAS", index=False)
        buff.seek(0)
        return Response(
            buff.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=SCAS_Estudiantes.xlsx"}
        )
    except Exception as e:
        return f"Error al generar Excel: {e}", 500
    
# Ruta para Resultado
@app.get('/resultado')
def resultado():
    uid = request.args.get('uid', type=int)
    if not uid:
        return "Falta el parámetro uid.", 400

    cn = get_db()
    cur = cn.cursor(dictionary=True)
    cur.execute("""
        SELECT 
               c.id_cuestionario, c.edad, c.genero, c.created_at,
               {", ".join([f"c.p{i}" for i in range(1,39)])}, 
               r.puntaje_Dim1, r.puntaje_Dim2, r.puntaje_Dim3,
               r.puntaje_Dim4, r.puntaje_Dim5, r.puntaje_Dim6,
               r.puntaje_total, r.nivel,
               u.nombre
        FROM (
            SELECT *
            FROM cuestionario
            WHERE id_usuario=%s
            ORDER BY created_at DESC
            LIMIT 1
        ) c
        JOIN usuario u        ON u.id_usuario = c.id_usuario
        LEFT JOIN resultado r ON r.id_cuestionario = c.id_cuestionario
    """, (uid,))
    row = cur.fetchone()
    cur.close(); cn.close()

    # Si no hay cuestionario o aún no hay fila en 'resultado'
    if not row or row.get('puntaje_total') is None:
        return render_template('resultado.html', notfound=True, uid=uid)

    sumas_dim = {
        "Dim1": row['puntaje_Dim1'],
        "Dim2": row['puntaje_Dim2'],
        "Dim3": row['puntaje_Dim3'],
        "Dim4": row['puntaje_Dim4'],
        "Dim5": row['puntaje_Dim5'],
        "Dim6": row['puntaje_Dim6'],
    }

    # features p1..p38 para ML
    respuestas = {f"p{i}": row.get(f"p{i}") for i in range(1, 39)} #NUEVO ML#
    pred_ml, proba_ml = ml_predict_from_answers(respuestas) #NUEVO ML#

    # Etiquetas por norma (para las chapitas de cada subescala)
    inter_sub, inter_total = interpreta_normas(
        row['genero'], row['edad'], sumas_dim, row['puntaje_total']
    )

    dims_order = ["Dim1","Dim2","Dim3","Dim4","Dim5","Dim6"]
    rows_view = []
    for d in dims_order:
        key = DIM_NOMBRES[d]
        rows_view.append({
            "code": key,
            "label": PRETTY[key],
            "score": sumas_dim[d],
            "level": inter_sub.get(key) or "-"
        })

    nivel_total = inter_total or row['nivel']

    return render_template(
        'resultado.html',
        notfound=False,
        uid=uid,
        nombre=row['nombre'],
        edad=row['edad'],
        rows=rows_view,
        total=row['puntaje_total'],
        nivel_total=nivel_total,
        pred_ml=pred_ml, #AGREGADO ML
        proba_ml=proba_ml #AGREGADO ML
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

        # >>> NUEVO: consultar rol y redirigir según sea admin o no
        cn = get_db()
        cur = cn.cursor()
        try:
            cur.execute("SELECT rol FROM usuario WHERE id_usuario=%s", (uid,))
            rol_row = cur.fetchone()
        finally:
            cur.close()
            cn.close()

        rol = (rol_row[0] if rol_row and rol_row[0] else '').lower()
        if rol == 'admin':
            return redirect(f'/form_panel?uid={uid}')              # admin -> panel
        else:
            return redirect(f'/cuestionario?uid={uid}') # estudiante/otros -> cuestionario

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
                SET edad=%s, genero=%s, {set_cols_p}
                WHERE id_cuestionario=%s
            """
            valores = [
                edad, genero,
                *[respuestas[f"p{i}"] for i in range(1, 39)],
                id_cuest
            ]
            cur.execute(sql, valores)
        else:
            # INSERT nuevo
            columnas_p = ", ".join([f"p{i}" for i in range(1, 39)])
            placeholders_p = ", ".join(["%s"] * 38)
            sql = f"""
                INSERT INTO cuestionario
                (id_usuario, edad, genero, {columnas_p})
                VALUES (%s, %s, %s, {placeholders_p})
            """
            valores = [
                id_usuario, edad, genero,
                *[respuestas[f"p{i}"] for i in range(1, 39)],
            ]
            cur.execute(sql, valores)
            id_cuest = cur.lastrowid

        # 6) INSERT o UPDATE en 'resultado' (como NO hay UNIQUE, lo controlamos por código)
        cur.execute("SELECT id_resultado FROM resultado WHERE id_cuestionario=%s LIMIT 1", (id_cuest,))
        row_res = cur.fetchone()

        if row_res:
            # UPDATE
            cur.execute("""
                UPDATE resultado
                SET puntaje_Dim1=%s, puntaje_Dim2=%s, puntaje_Dim3=%s,
                    puntaje_Dim4=%s, puntaje_Dim5=%s, puntaje_Dim6=%s,
                    puntaje_total=%s, nivel=%s
                WHERE id_cuestionario=%s
            """, (
                sumas["Dim1"], sumas["Dim2"], sumas["Dim3"],
                sumas["Dim4"], sumas["Dim5"], sumas["Dim6"],
                puntaje_total, nivel_txt,
                id_cuest
            ))
        else:
            # INSERT
            cur.execute("""
                INSERT INTO resultado(
                  id_cuestionario, puntaje_Dim1, puntaje_Dim2, puntaje_Dim3,
                  puntaje_Dim4, puntaje_Dim5, puntaje_Dim6, puntaje_total, nivel
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                id_cuest,
                sumas["Dim1"], sumas["Dim2"], sumas["Dim3"],
                sumas["Dim4"], sumas["Dim5"], sumas["Dim6"],
                puntaje_total, nivel_txt
            ))

        cn.commit()
        cur.close(); cn.close()

        # 7) Redirigir al resultado
        return redirect(f"/resultado?uid={id_usuario}")

    except Exception as e:
        return f"Error al guardar: {e}", 400
        


# === 9) Run ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))   # Render define PORT; 5000 de fallback local
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
