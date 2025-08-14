# scas/rules.py
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

# === 4) Helpers ===
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