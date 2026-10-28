"""Contrato de datos y extracción de características del banco hidráulico.

Cada ciclo de 60 s trae 43 680 lecturas repartidas en 17 sensores con frecuencias de 1, 10
y 100 Hz. Este módulo las resume en una tabla de una fila por ciclo, que es lo que ven
tanto el baseline como los modelos.

Decisiones de diseño, todas declaradas antes de medir:

- **Objetivo: la válvula.** Ningún sensor individual la delata (correlación máxima 0.231),
  de modo que es una anomalía contextual real. El refrigerador se descartó porque TS1 sola
  correlaciona -0.911 con su estado: un operador lo vería en el termómetro.
- **Solo ciclos estables.** Los 1080 ciclos con válvula degradada son todos estables,
  mientras que de los 1125 óptimos solo 369 lo son. Incluir los transitorios permitiría
  aprender «transitorio implica sano» y acertar 756 casos sin entender nada.
- **Partición estratificada, no cronológica.** El archivo está ordenado por el estado del
  refrigerador (correlación del índice con ese estado: +0.884), así que un corte temporal
  entrenaría con un solo régimen de refrigeración.
- **Sensores virtuales marcados.** CE, CP y SE no se miden: se calculan. Se conservan, pero
  `SENSORES_VIRTUALES` permite la ablación que comprueba que el resultado no depende de
  ellos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CRUDO = RAIZ / "data" / "crudo"

SENSORES: dict[str, tuple[int, str]] = {
    "PS1": (100, "bar"), "PS2": (100, "bar"), "PS3": (100, "bar"),
    "PS4": (100, "bar"), "PS5": (100, "bar"), "PS6": (100, "bar"),
    "EPS1": (100, "W"),
    "FS1": (10, "l/min"), "FS2": (10, "l/min"),
    "TS1": (1, "degC"), "TS2": (1, "degC"), "TS3": (1, "degC"), "TS4": (1, "degC"),
    "VS1": (1, "mm/s"),
    "CE": (1, "%"), "CP": (1, "kW"), "SE": (1, "%"),
}
SENSORES_VIRTUALES = ("CE", "CP", "SE")

# Cuántos tramos temporales se resumen por sensor. Los sensores rápidos (100 y 10 Hz)
# llevan seis; los lentos, muestreados a 1 Hz, solo tres, porque un ciclo son 60 lecturas.
SEGMENTOS_RAPIDOS = 6
SEGMENTOS_LENTOS = 3
SENSORES_RAPIDOS = tuple(s for s, (hz, _) in SENSORES.items() if hz >= 10)
SENSORES_LENTOS = tuple(s for s in SENSORES if s not in SENSORES_RAPIDOS)
AGREGADOS = ("media", "desv")

COLUMNAS_PERFIL = ("refrigerador", "valvula", "fuga_bomba", "acumulador", "estable")
VALVULA_OPTIMA = 100
VALVULA_NIVELES = {100: "optimo", 90: "retardo pequeno", 80: "retardo severo",
                   73: "cerca del fallo total"}
ACUMULADOR_NIVELES = {130: "optimo", 115: "presion ligeramente reducida",
                      100: "presion severamente reducida", 90: "cerca del fallo total"}
FUGA_NIVELES = {0: "sin fuga", 1: "fuga debil", 2: "fuga severa"}

# Los tres objetivos del trabajo, elegidos por dificultad creciente de deteccion con un
# solo sensor. El refrigerador queda fuera: CE lo predice con 0.992 y TS1 con -0.911, de
# modo que un operador lo veria en una sola aguja y no sirve para la pregunta.
#
#   valvula     correlacion maxima de un sensor 0.231  -> anomalia contextual pura
#   fuga_bomba  correlacion maxima de un sensor 0.469  -> caso intermedio, y el componente
#                                                         lo acciona el motor electrico
#   acumulador  correlacion maxima de un sensor 0.241  -> el mas complejo segun la
#                                                         documentacion del conjunto
OBJETIVOS: dict[str, dict] = {
    "valvula": {"optimo": 100, "niveles": VALVULA_NIVELES,
                "descripcion": "retardo de conmutacion de la valvula"},
    "acumulador": {"optimo": 130, "niveles": ACUMULADOR_NIVELES,
                   "descripcion": "presion del acumulador hidraulico"},
    "fuga_bomba": {"optimo": 0, "niveles": FUGA_NIVELES,
                   "descripcion": "fuga interna de la bomba accionada por el motor"},
}
OBJETIVO_POR_DEFECTO = "valvula"


def resumir_sensor(matriz: np.ndarray, segmentos: int) -> dict[str, np.ndarray]:
    """Media y desviación del ciclo entero, más la media de cada tramo temporal.

    La media del ciclo es lo que mostraría una aguja del panel. Los tramos añaden **cuándo**
    ocurre cada cosa dentro de los 60 segundos, y eso resultó ser lo decisivo: la válvula es
    un componente de conmutación, de modo que un retardo cambia el momento en que sube la
    presión, no su promedio. Con solo los agregados del ciclo, la detección de la válvula
    cae a ROC-AUC 0.58; con seis tramos sube a 0.996.
    """
    resumen = {"media": matriz.mean(axis=1), "desv": matriz.std(axis=1)}
    for i, tramo in enumerate(np.array_split(matriz, segmentos, axis=1), start=1):
        resumen[f"t{i}"] = tramo.mean(axis=1)
    return resumen


def construir_tabla(crudo: Path = CRUDO) -> pd.DataFrame:
    """Lee los 17 archivos de sensor y el perfil, y devuelve una fila por ciclo."""
    perfil = np.loadtxt(crudo / "profile.txt")
    if perfil.shape[1] != len(COLUMNAS_PERFIL):
        raise ValueError(f"profile.txt debería tener {len(COLUMNAS_PERFIL)} columnas")

    columnas: dict[str, np.ndarray] = {}
    for sensor in SENSORES:
        matriz = np.loadtxt(crudo / f"{sensor}.txt")
        if matriz.shape[0] != perfil.shape[0]:
            raise ValueError(f"{sensor}.txt tiene {matriz.shape[0]} ciclos, "
                             f"el perfil tiene {perfil.shape[0]}")
        segmentos = (SEGMENTOS_RAPIDOS if sensor in SENSORES_RAPIDOS
                     else SEGMENTOS_LENTOS)
        for nombre, valores in resumir_sensor(matriz, segmentos).items():
            columnas[f"{sensor}_{nombre}"] = valores

    tabla = pd.DataFrame(columnas)
    tabla.insert(0, "ciclo", np.arange(len(tabla)))
    for i, nombre in enumerate(COLUMNAS_PERFIL):
        tabla[nombre] = perfil[:, i]
    return etiquetar(tabla, OBJETIVO_POR_DEFECTO)


def etiquetar(tabla: pd.DataFrame, objetivo: str) -> pd.DataFrame:
    """Añade `anomalia` y `severidad` para el componente elegido.

    El mismo conjunto de ciclos sirve para los dos objetivos: lo único que cambia es qué
    componente se considera el fallo y cuál es su estado óptimo.
    """
    if objetivo not in OBJETIVOS:
        raise ValueError(f"Objetivo desconocido: {objetivo}. "
                         f"Disponibles: {sorted(OBJETIVOS)}")
    definicion = OBJETIVOS[objetivo]
    tabla = tabla.copy()
    tabla["objetivo"] = objetivo
    tabla["anomalia"] = (tabla[objetivo] != definicion["optimo"]).astype(int)
    tabla["severidad"] = tabla[objetivo].map(definicion["niveles"])
    return tabla


def solo_estables(tabla: pd.DataFrame) -> pd.DataFrame:
    """Descarta los ciclos que pueden no haber alcanzado régimen estacionario."""
    return tabla[tabla.estable == 0].reset_index(drop=True)


def columnas_sensor(tabla: pd.DataFrame, incluir_virtuales: bool = True) -> list[str]:
    virtuales = () if incluir_virtuales else SENSORES_VIRTUALES
    return [c for c in tabla.columns
            if "_" in c and c.split("_")[0] in SENSORES
            and c.split("_")[0] not in virtuales]


def columnas_aguja(tabla: pd.DataFrame, incluir_virtuales: bool = True) -> list[str]:
    """Las 17 medias por ciclo: lo que un operador leería en el panel."""
    return [c for c in columnas_sensor(tabla, incluir_virtuales) if c.endswith("_media")]


def validar_contrato(tabla: pd.DataFrame) -> None:
    """Comprueba lo que el resto del proyecto da por supuesto."""
    faltan = set(COLUMNAS_PERFIL) - set(tabla.columns)
    if faltan:
        raise ValueError(f"Faltan columnas del perfil: {sorted(faltan)}")
    esperadas = (len(SENSORES) * len(AGREGADOS)
                 + len(SENSORES_RAPIDOS) * SEGMENTOS_RAPIDOS
                 + len(SENSORES_LENTOS) * SEGMENTOS_LENTOS)
    hay = len(columnas_sensor(tabla))
    if hay != esperadas:
        raise ValueError(f"Se esperaban {esperadas} columnas de sensor y hay {hay}")
    if tabla[columnas_sensor(tabla)].isna().any().any():
        raise ValueError("Hay valores ausentes en las columnas de sensor")
    if not set(tabla.anomalia.unique()) <= {0, 1}:
        raise ValueError("La etiqueta de anomalía debe ser binaria")
    for objetivo, definicion in OBJETIVOS.items():
        vistos = set(tabla[objetivo].unique())
        if not vistos <= set(definicion["niveles"]):
            raise ValueError(f"Niveles inesperados en {objetivo}: {sorted(vistos)}")


def particion(tabla: pd.DataFrame, fraccion_prueba: float = 0.3, semilla: int = 42,
              objetivo: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición estratificada por el nivel de degradación del objetivo.

    No es cronológica a propósito: el archivo está ordenado por estado del refrigerador,
    de modo que un corte temporal confundiría el régimen de refrigeración con el objetivo.
    """
    from sklearn.model_selection import train_test_split

    columna = objetivo or (tabla.objetivo.iloc[0] if "objetivo" in tabla.columns
                           else OBJETIVO_POR_DEFECTO)
    entrena, prueba = train_test_split(tabla, test_size=fraccion_prueba, random_state=semilla,
                                       stratify=tabla[columna])
    return entrena.reset_index(drop=True), prueba.reset_index(drop=True)


def solo_normales(tabla: pd.DataFrame) -> pd.DataFrame:
    """Los ciclos con válvula óptima: lo único con lo que se entrena un detector."""
    return tabla[tabla.anomalia == 0].reset_index(drop=True)
