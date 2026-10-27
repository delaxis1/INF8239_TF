"""El baseline de agujas y las métricas tienen que comportarse como se declaró."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inf8239_tf.baseline import AlarmaPorAguja
from inf8239_tf.evaluacion import (deteccion_por_severidad, linea_de_azar, precision_en_k,
                                   umbral_por_falsas_alarmas)

COLUMNAS = ["A_media", "B_media"]


@pytest.fixture
def normales() -> pd.DataFrame:
    generador = np.random.default_rng(42)
    return pd.DataFrame({"A_media": generador.normal(10, 1, 400),
                         "B_media": generador.normal(50, 5, 400)})


def test_un_ciclo_en_el_centro_no_dispara(normales):
    alarma = AlarmaPorAguja.ajustar(normales, COLUMNAS)
    centro = pd.DataFrame([alarma.centro])
    assert alarma.puntuar(centro)[0] == pytest.approx(0.0, abs=1e-9)
    assert alarma.alarma(centro)[0] == 0


def test_el_limite_vale_exactamente_uno(normales):
    """La puntuación está normalizada para que 1 sea justo el borde de la zona roja."""
    alarma = AlarmaPorAguja.ajustar(normales, COLUMNAS)
    borde = pd.DataFrame([alarma.superior])
    assert alarma.puntuar(borde)[0] == pytest.approx(1.0)


def test_una_aguja_fuera_de_banda_dispara(normales):
    alarma = AlarmaPorAguja.ajustar(normales, COLUMNAS)
    fuera = pd.DataFrame([{"A_media": alarma.superior["A_media"] * 2,
                           "B_media": alarma.centro["B_media"]}])
    assert alarma.puntuar(fuera)[0] > 1.0
    assert alarma.alarma(fuera)[0] == 1
    assert alarma.aguja_culpable(fuera).iloc[0] == "A_media"


def test_el_baseline_es_ciego_a_lo_contextual(normales):
    """Dos lecturas normales por separado pero imposibles juntas no disparan la alarma.

    Es la limitación que define al baseline, y la razón del trabajo. Aquí se fija como
    prueba para que nadie la confunda con un error.
    """
    alarma = AlarmaPorAguja.ajustar(normales, COLUMNAS)
    combinacion_rara = pd.DataFrame([{
        "A_media": normales.A_media.quantile(0.99),
        "B_media": normales.B_media.quantile(0.01),
    }])
    assert alarma.alarma(combinacion_rara)[0] == 0


def test_el_umbral_respeta_la_tasa_de_falsas_alarmas():
    puntuacion = np.linspace(0.0, 1.0, 1000)
    umbral = umbral_por_falsas_alarmas(puntuacion, 0.05)
    assert np.mean(puntuacion > umbral) == pytest.approx(0.05, abs=0.01)


def test_una_tasa_imposible_se_rechaza():
    with pytest.raises(ValueError):
        umbral_por_falsas_alarmas(np.array([0.1, 0.2]), 1.5)


def test_la_linea_de_azar_es_la_prevalencia():
    assert linea_de_azar(np.array([1, 1, 1, 0])) == pytest.approx(0.75)


def test_precision_en_k_mira_solo_los_k_peores():
    y = np.array([0, 1, 1, 0])
    puntuacion = np.array([0.1, 0.9, 0.8, 0.2])
    assert precision_en_k(y, puntuacion, 2) == pytest.approx(1.0)
    assert precision_en_k(y, puntuacion, 4) == pytest.approx(0.5)


def test_la_deteccion_por_severidad_cubre_todos_los_niveles():
    tabla = pd.DataFrame({
        "objetivo": ["valvula"] * 4,
        "valvula": [100, 100, 90, 73],
        "severidad": ["optimo", "optimo", "retardo pequeno", "cerca del fallo total"],
    })
    resultado = deteccion_por_severidad(tabla, np.array([0.1, 0.2, 2.0, 3.0]), umbral=1.0)
    assert list(resultado["nivel"]) == [100, 90, 73]
    assert list(resultado["tasa"]) == [0.0, 1.0, 1.0]


def test_la_severidad_funciona_igual_con_el_otro_objetivo():
    """El mismo código sirve para la válvula y para el acumulador."""
    tabla = pd.DataFrame({
        "objetivo": ["acumulador"] * 3,
        "acumulador": [130, 115, 90],
        "severidad": ["optimo", "presion ligeramente reducida", "cerca del fallo total"],
    })
    resultado = deteccion_por_severidad(tabla, np.array([0.1, 2.0, 3.0]), umbral=1.0)
    assert list(resultado["nivel"]) == [130, 115, 90]
    assert list(resultado["tasa"]) == [0.0, 1.0, 1.0]
