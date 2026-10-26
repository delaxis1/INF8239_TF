"""Contrato de datos y prevención de fuga.

Cada prueba fija una decisión que el resto del proyecto da por supuesta. Si alguien cambia
la extracción o la partición y rompe un supuesto, estas pruebas lo dicen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inf8239_tf.data import (RAIZ, SENSORES, SENSORES_VIRTUALES, VALVULA_NIVELES,
                             VALVULA_OPTIMA, columnas_aguja, columnas_sensor, particion,
                             resumir_sensor, solo_estables, solo_normales, validar_contrato)

CICLOS = RAIZ / "data" / "ciclos.csv"


@pytest.fixture(scope="module")
def tabla() -> pd.DataFrame:
    if not CICLOS.exists():
        pytest.skip("Ejecuta scripts/download_data.py y scripts/build_features.py primero")
    return pd.read_csv(CICLOS)


def test_el_contrato_se_cumple(tabla):
    validar_contrato(tabla)


def test_hay_cinco_estadisticos_por_sensor(tabla):
    assert len(columnas_sensor(tabla)) == len(SENSORES) * 5
    assert len(columnas_aguja(tabla)) == len(SENSORES)


def test_la_tabla_guardada_solo_tiene_ciclos_estables(tabla):
    assert (tabla.estable == 0).all()
    assert len(tabla) == 1449


def test_la_pendiente_recupera_una_rampa_conocida():
    """Una rampa de 0 a 3 sobre tiempo normalizado tiene pendiente 3."""
    rampa = np.linspace(0.0, 3.0, 60).reshape(1, -1)
    assert resumir_sensor(rampa)["pendiente"][0] == pytest.approx(3.0)
    assert resumir_sensor(np.full((1, 60), 7.0))["pendiente"][0] == pytest.approx(0.0)


def test_el_indicador_de_estabilidad_no_es_una_variable(tabla):
    """`estable` delata la válvula: de los degradados, el 100 % son estables.

    Nunca puede entrar como variable predictora, y por eso no está entre las columnas de
    sensor. Se usa solo para filtrar, antes de modelar.
    """
    assert "estable" not in columnas_sensor(tabla)
    assert "estable" not in columnas_aguja(tabla)


def test_las_etiquetas_del_perfil_no_son_variables(tabla):
    for etiqueta in ("valvula", "refrigerador", "fuga_bomba", "acumulador", "anomalia",
                     "severidad", "ciclo"):
        assert etiqueta not in columnas_sensor(tabla)


def test_la_ablacion_quita_los_sensores_virtuales(tabla):
    con = columnas_sensor(tabla, incluir_virtuales=True)
    sin = columnas_sensor(tabla, incluir_virtuales=False)
    assert len(con) - len(sin) == len(SENSORES_VIRTUALES) * 5
    assert not any(c.split("_")[0] in SENSORES_VIRTUALES for c in sin)


def test_la_anomalia_es_toda_valvula_no_optima(tabla):
    assert (tabla.loc[tabla.valvula == VALVULA_OPTIMA, "anomalia"] == 0).all()
    assert (tabla.loc[tabla.valvula != VALVULA_OPTIMA, "anomalia"] == 1).all()
    assert set(tabla.valvula.unique()) == set(VALVULA_NIVELES)


def test_la_particion_es_estratificada_y_disjunta(tabla):
    entrena, prueba = particion(tabla)
    assert set(entrena.ciclo) & set(prueba.ciclo) == set()
    assert len(entrena) + len(prueba) == len(tabla)
    for nivel in VALVULA_NIVELES:
        esperado = (tabla.valvula == nivel).mean()
        assert (entrena.valvula == nivel).mean() == pytest.approx(esperado, abs=0.02)
        assert (prueba.valvula == nivel).mean() == pytest.approx(esperado, abs=0.02)


def test_el_detector_solo_se_entrena_con_ciclos_normales(tabla):
    entrena, _ = particion(tabla)
    normales = solo_normales(entrena)
    assert (normales.anomalia == 0).all()
    assert (normales.valvula == VALVULA_OPTIMA).all()
    assert len(normales) >= 200, "Quedan muy pocos ciclos normales para entrenar"


def test_solo_estables_no_pierde_ninguna_anomalia():
    """El filtro de estabilidad descarta transitorios, que son todos de válvula óptima."""
    sintetica = pd.DataFrame({"estable": [0, 1, 0, 1], "anomalia": [1, 0, 1, 0],
                              "valvula": [90, 100, 73, 100]})
    filtrada = solo_estables(sintetica)
    assert len(filtrada) == 2
    assert filtrada.anomalia.sum() == sintetica.anomalia.sum()
