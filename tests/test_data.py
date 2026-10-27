"""Contrato de datos y prevención de fuga.

Cada prueba fija una decisión que el resto del proyecto da por supuesta. Si alguien cambia
la extracción o la partición y rompe un supuesto, estas pruebas lo dicen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inf8239_tf.data import (OBJETIVOS, RAIZ, SEGMENTOS_LENTOS, SEGMENTOS_RAPIDOS, SENSORES,
                             SENSORES_LENTOS, SENSORES_RAPIDOS, SENSORES_VIRTUALES,
                             VALVULA_NIVELES, VALVULA_OPTIMA, columnas_aguja,
                             columnas_sensor, etiquetar, particion, resumir_sensor,
                             solo_estables,
                             solo_normales, validar_contrato)

CICLOS = RAIZ / "data" / "ciclos.csv"


@pytest.fixture(scope="module")
def tabla() -> pd.DataFrame:
    if not CICLOS.exists():
        pytest.skip("Ejecuta scripts/download_data.py y scripts/build_features.py primero")
    return pd.read_csv(CICLOS)


def test_el_contrato_se_cumple(tabla):
    validar_contrato(tabla)


def test_el_numero_de_columnas_cuadra(tabla):
    esperadas = (len(SENSORES) * 2 + len(SENSORES_RAPIDOS) * SEGMENTOS_RAPIDOS
                 + len(SENSORES_LENTOS) * SEGMENTOS_LENTOS)
    assert len(columnas_sensor(tabla)) == esperadas == 112
    assert len(columnas_aguja(tabla)) == len(SENSORES)


def test_la_tabla_guardada_solo_tiene_ciclos_estables(tabla):
    assert (tabla.estable == 0).all()
    assert len(tabla) == 1449


def test_los_tramos_recuperan_un_perfil_conocido():
    """Los tramos capturan CUANDO pasa cada cosa, que es donde vive el estado de la valvula."""
    escalon = np.concatenate([np.zeros(30), np.ones(30)]).reshape(1, -1)
    resumen = resumir_sensor(escalon, segmentos=6)
    assert [round(float(resumen[f"t{i}"][0]), 3) for i in range(1, 7)] == [0, 0, 0, 1, 1, 1]
    assert resumen["media"][0] == pytest.approx(0.5)

    # El mismo escalon desplazado: misma media, tramos distintos. Eso es el retardo.
    tarde = np.concatenate([np.zeros(40), np.ones(20)]).reshape(1, -1)
    assert resumir_sensor(tarde, 6)["media"][0] != pytest.approx(resumen["media"][0])
    assert [round(float(resumir_sensor(tarde, 6)[f"t{i}"][0]), 3)
            for i in range(1, 7)] == [0, 0, 0, 0, 1, 1]


def test_el_indicador_de_estabilidad_no_es_una_variable(tabla):
    """`estable` delata la válvula: de los degradados, el 100 % son estables.

    Nunca puede entrar como variable predictora, y por eso no está entre las columnas de
    sensor. Se usa solo para filtrar, antes de modelar.
    """
    assert "estable" not in columnas_sensor(tabla)
    assert "estable" not in columnas_aguja(tabla)


def test_los_sensores_rapidos_llevan_mas_tramos(tabla):
    """Los de 1 Hz solo tienen 60 lecturas por ciclo: seis tramos serian ruido."""
    assert set(SENSORES_RAPIDOS) & set(SENSORES_LENTOS) == set()
    assert len(SENSORES_RAPIDOS) + len(SENSORES_LENTOS) == len(SENSORES)
    rapido = [c for c in tabla.columns if c.startswith("PS1_t")]
    lento = [c for c in tabla.columns if c.startswith("TS1_t")]
    assert len(rapido) == SEGMENTOS_RAPIDOS
    assert len(lento) == SEGMENTOS_LENTOS


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


def test_los_dos_objetivos_se_etiquetan_desde_la_misma_tabla(tabla):
    """Válvula y acumulador comparten ciclos: solo cambia qué se considera el fallo."""
    for objetivo, definicion in OBJETIVOS.items():
        etiquetada = etiquetar(tabla, objetivo)
        assert etiquetada.objetivo.unique().tolist() == [objetivo]
        assert (etiquetada.loc[etiquetada[objetivo] == definicion["optimo"],
                               "anomalia"] == 0).all()
        assert (etiquetada.loc[etiquetada[objetivo] != definicion["optimo"],
                               "anomalia"] == 1).all()
        assert etiquetada.severidad.notna().all()


def test_un_objetivo_inexistente_se_rechaza(tabla):
    with pytest.raises(ValueError, match="Objetivo desconocido"):
        etiquetar(tabla, "turbina")


def test_la_particion_estratifica_por_el_objetivo_elegido(tabla):
    acumulador = etiquetar(tabla, "acumulador")
    entrena, prueba = particion(acumulador, objetivo="acumulador")
    for nivel in OBJETIVOS["acumulador"]["niveles"]:
        esperado = (acumulador.acumulador == nivel).mean()
        assert (prueba.acumulador == nivel).mean() == pytest.approx(esperado, abs=0.02)
