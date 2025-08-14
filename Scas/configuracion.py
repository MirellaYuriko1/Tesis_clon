# --- Librerías y conexión MySQL ---
import os
import mysql.connector
from dotenv import load_dotenv
#------------------------------

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

