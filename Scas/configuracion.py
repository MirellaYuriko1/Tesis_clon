# --- Librerías y conexión MySQL ---
from dotenv import load_dotenv
from pathlib import Path
import os, mysql.connector
#------------------------------

#----------------------------------------------
# 1) Cargar variables del .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

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

