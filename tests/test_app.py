"""El tablero arranca sin navegador y declara sus limites."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from inf8239_tf.data import RAIZ, columnas_sensor, etiquetar, particion, solo_normales
from inf8239_tf.modelos import Mahalanobis

APP = Path(__file__).parents[1] / "app.py"
CICLOS = RAIZ / "data" / "ciclos.csv"


@pytest.fixture(scope="module")
def tabla() -> pd.DataFrame:
    if not CICLOS.exists():
        pytest.skip("Ejecuta scripts/build_features.py primero")
    return etiquetar(pd.read_csv(CICLOS), "valvula")


def test_el_tablero_arranca_y_avisa_que_no_sustituye_la_decision():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    texto = " ".join(w.value.lower() for w in at.warning)
    assert "no la sustituye" in texto
    assert "banco de pruebas" in texto


def test_el_tablero_publica_sus_limites():
    at = AppTest.from_file(APP, default_timeout=180).run()
    cuerpo = " ".join(m.value.lower() for m in at.markdown)
    assert "no permite concluir" in " ".join(s.value.lower() for s in at.subheader)
    assert "prevalencia" in cuerpo
    assert "marcar no es diagnosticar" in cuerpo


def test_las_contribuciones_suman_la_distancia_al_cuadrado(tabla):
    """La explicacion por variable tiene que repartir exactamente la puntuacion."""
    entrena, prueba = particion(tabla, objetivo="valvula")
    ajuste = solo_normales(entrena).iloc[:150]
    maha = Mahalanobis(columnas_sensor(tabla)).ajustar(ajuste)
    muestra = prueba.head(20)
    assert np.allclose(maha.contribuciones(muestra).sum(axis=1).to_numpy(),
                       maha.puntuar(muestra) ** 2, rtol=1e-6)


def test_las_variables_principales_vienen_ordenadas(tabla):
    entrena, prueba = particion(tabla, objetivo="valvula")
    maha = Mahalanobis(columnas_sensor(tabla)).ajustar(solo_normales(entrena).iloc[:150])
    for fila in maha.principales(prueba.head(5), cuantas=4):
        aportes = [valor for _, valor in fila]
        assert len(fila) == 4
        assert aportes == sorted(aportes, reverse=True)
