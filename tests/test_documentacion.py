"""Las cifras del README y del Model Card tienen que salir de `reports/`.

Un informe se desincroniza en silencio en cuanto alguien vuelve a ejecutar un script y no
actualiza el texto. Estas pruebas leen las cifras de los artefactos y comprueban que
aparecen literalmente en la documentación, de modo que una ejecución nueva que cambie un
resultado rompe la suite en lugar de dejar un documento que miente.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
REPORTES = RAIZ / "reports"
README = RAIZ / "README.md"
MODEL_CARD = RAIZ / "docs" / "MODEL_CARD.md"


def comparacion(objetivo: str) -> pd.DataFrame:
    ruta = REPORTES / f"modelos_{objetivo}" / "comparacion.csv"
    if not ruta.exists():
        pytest.skip(f"Ejecuta scripts/run_models.py --objetivo {objetivo} primero")
    return pd.read_csv(ruta).set_index("alternativa")


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def model_card() -> str:
    return MODEL_CARD.read_text(encoding="utf-8")


@pytest.mark.parametrize("alternativa", ["baseline_agujas", "mahalanobis", "vae"])
def test_el_readme_cita_el_pr_auc_medido(readme, alternativa):
    valor = comparacion("valvula").loc[alternativa, "pr_auc"]
    assert f"{valor:.4f}" in readme, (
        f"El README no contiene el PR-AUC medido de {alternativa}: {valor:.4f}")


@pytest.mark.parametrize("alternativa", ["baseline_agujas", "mahalanobis", "vae"])
def test_el_readme_cita_el_roc_auc_medido(readme, alternativa):
    valor = comparacion("valvula").loc[alternativa, "roc_auc"]
    assert f"{valor:.4f}" in readme


def test_la_linea_de_azar_esta_declarada(readme):
    azar = comparacion("valvula").linea_de_azar.iloc[0]
    assert f"{azar:.3f}" in readme, (
        "El README debe declarar la línea de azar del PR-AUC: sin ella un 0.80 parece "
        "bueno y no lo es")


@pytest.mark.parametrize("objetivo", ["valvula", "acumulador"])
@pytest.mark.parametrize("alternativa", ["baseline_agujas", "mahalanobis", "vae"])
def test_el_model_card_cita_las_dos_tablas(model_card, objetivo, alternativa):
    valor = comparacion(objetivo).loc[alternativa, "pr_auc"]
    assert f"{valor:.4f}" in model_card


def test_la_deteccion_por_severidad_del_readme_coincide():
    ruta = REPORTES / "modelos_valvula" / "severidad_comparada.csv"
    if not ruta.exists():
        pytest.skip("Ejecuta scripts/run_models.py primero")
    severidad = pd.read_csv(ruta)
    texto = README.read_text(encoding="utf-8")
    degradados = severidad[severidad.nivel != 100]
    for fila in degradados.itertuples():
        assert f"{fila.baseline_agujas:.1%}".replace("%", " %") in texto or \
               f"{fila.baseline_agujas * 100:.1f} %" in texto, (
            f"Falta en el README la detección del baseline en «{fila.severidad}»")


def test_el_limite_por_regimen_del_readme_coincide():
    ruta = REPORTES / "decision_acumulador" / "regimen_refrigerador.csv"
    if not ruta.exists():
        pytest.skip("Ejecuta scripts/run_decision.py --objetivo acumulador primero")
    regimen = pd.read_csv(ruta)
    texto = README.read_text(encoding="utf-8")
    for fila in regimen.itertuples():
        assert f"{fila.recall:.3f}" in texto, (
            f"Falta en el README el recall con refrigerador al {int(fila.refrigerador)} %")


def test_la_cita_del_conjunto_lleva_licencia_doi_y_paginas(readme):
    """Verificado contra la página de UCI y el registro del artículo."""
    assert "CC BY 4.0" in readme
    assert "10.24432/C5CW21" in readme
    assert "10.1109/I2MTC.2015.7151267" in readme
    assert "210-215" in readme, "Las páginas del artículo de Helwig son 210-215"
