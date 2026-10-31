"""Las columnas de coste de `reports/robustez/` tienen que medir lo que dicen medir.

`run_seeds.py` arrancaba el cronómetro **después** de ajustar los tres detectores, así que
la columna `ajuste_s` guardaba el tiempo de puntuación: el VAE aparecía ajustándose en
0.006 s cuando en `modelos_*/comparacion.csv` tardaba casi ocho segundos. Y `tamano_kb`
estaba fijo en 0.0 para los tres.

Nada del informe dependía de esas cifras, pero una columna que dice una cosa y mide otra es
exactamente lo que nadie revisa. Estas pruebas fijan la convención: el reloj cubre solo el
ajuste, el tamaño sale del número de parámetros, y las dos cosas significan lo mismo en los
dos sitios donde se miden.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
REPORTES = RAIZ / "reports"
POR_SEMILLA = REPORTES / "robustez" / "por_semilla.csv"
COSTE = ("ajuste_s", "tamano_kb", "parametros")


@pytest.fixture(scope="module")
def semillas() -> pd.DataFrame:
    if not POR_SEMILLA.exists():
        pytest.skip("Ejecuta scripts/run_seeds.py primero")
    return pd.read_csv(POR_SEMILLA)


def test_estan_las_tres_columnas_de_coste(semillas: pd.DataFrame) -> None:
    faltan = [c for c in COSTE if c not in semillas.columns]
    assert not faltan, f"por_semilla.csv no publica el coste: {faltan}"


def test_el_vae_tarda_mucho_mas_en_ajustarse_que_mahalanobis(semillas: pd.DataFrame) -> None:
    """La prueba que habría cazado el error: entrenar una red no cuesta lo mismo que
    invertir una covarianza, y si el CSV dice que sí, el cronómetro está mal puesto."""
    for objetivo, grupo in semillas.groupby("objetivo"):
        por_alternativa = grupo.groupby("alternativa").ajuste_s.mean()
        assert por_alternativa["vae"] > 10 * por_alternativa["mahalanobis"], (
            f"{objetivo}: el VAE dice ajustarse en {por_alternativa['vae']:.4f} s y "
            f"Mahalanobis en {por_alternativa['mahalanobis']:.4f} s")


def test_el_tamano_no_es_cero_donde_hay_modelo(semillas: pd.DataFrame) -> None:
    con_pesos = semillas[semillas.alternativa.isin(["mahalanobis", "vae"])]
    assert (con_pesos.tamano_kb > 0).all()
    assert (semillas.parametros > 0).all()


def test_el_tamano_concuerda_con_los_parametros(semillas: pd.DataFrame) -> None:
    """float64 en la covarianza de Mahalanobis, float32 en los pesos del VAE."""
    anchos = {"mahalanobis": 8, "vae": 4}
    for alternativa, ancho in anchos.items():
        filas = semillas[semillas.alternativa == alternativa]
        esperado = (filas.parametros * ancho / 1024).round(2)
        assert (filas.tamano_kb - esperado).abs().max() < 0.01, alternativa


@pytest.mark.parametrize("objetivo", ["valvula", "acumulador"])
def test_los_parametros_coinciden_con_el_experimento_principal(semillas: pd.DataFrame,
                                                               objetivo: str) -> None:
    """El número de parámetros depende de las variables, no de la partición: tiene que ser
    el mismo en `robustez/` y en `modelos_<objetivo>/`, o las dos medidas no son
    comparables entre sí."""
    ruta = REPORTES / f"modelos_{objetivo}" / "comparacion.csv"
    if not ruta.exists():
        pytest.skip(f"Ejecuta scripts/run_models.py --objetivo {objetivo} primero")
    principal = pd.read_csv(ruta).set_index("alternativa").parametros
    for alternativa, grupo in semillas[semillas.objetivo == objetivo].groupby("alternativa"):
        assert grupo.parametros.nunique() == 1, f"{alternativa} cambia entre semillas"
        assert int(grupo.parametros.iloc[0]) == int(principal[alternativa]), alternativa
