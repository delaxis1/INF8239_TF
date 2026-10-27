"""Umbral por costo, capacidad y política de abstención."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inf8239_tf.decision import (Costos, Politica, auditoria_por_regimen,
                                 resumen_de_politica, sensibilidad_al_costo,
                                 umbral_por_capacidad, umbral_por_costo)


def test_el_costo_pesa_mas_los_fallos_no_detectados():
    y = np.array([1, 1, 0, 0])
    un_falso_negativo = np.array([True, False, False, False])
    un_falso_positivo = np.array([True, True, True, False])
    costos = Costos(fallo_no_detectado=10.0)
    assert costos.total(y, un_falso_negativo) == 10.0
    assert costos.total(y, un_falso_positivo) == 1.0


def test_el_umbral_por_costo_minimiza_el_costo():
    y = np.array([0] * 50 + [1] * 50)
    puntuacion = np.concatenate([np.random.default_rng(0).normal(0, 1, 50),
                                 np.random.default_rng(1).normal(5, 1, 50)])
    umbral = umbral_por_costo(y, puntuacion, Costos(fallo_no_detectado=5.0))
    marcados = puntuacion > umbral
    assert marcados[y == 1].mean() > 0.9
    assert marcados[y == 0].mean() < 0.2


def test_cuanto_mas_caro_el_fallo_mas_bajo_el_umbral():
    """Si no detectar es carísimo, conviene marcar más, no menos."""
    y = np.array([0] * 60 + [1] * 40)
    puntuacion = np.concatenate([np.random.default_rng(2).normal(0, 1, 60),
                                 np.random.default_rng(3).normal(2, 1, 40)])
    tabla = sensibilidad_al_costo(y, puntuacion, (1.0, 10.0, 100.0))
    assert list(tabla.razon_fn_fa) == [1.0, 10.0, 100.0]
    assert tabla.umbral.is_monotonic_decreasing
    assert tabla.recall.is_monotonic_increasing


def test_la_capacidad_marca_la_fraccion_pedida():
    puntuacion = np.linspace(0.0, 1.0, 1000)
    umbral = umbral_por_capacidad(puntuacion, 0.15)
    assert np.mean(puntuacion > umbral) == pytest.approx(0.15, abs=0.01)


def test_una_capacidad_imposible_se_rechaza():
    with pytest.raises(ValueError):
        umbral_por_capacidad(np.array([0.1, 0.2]), 0.0)


def test_la_politica_ordena_sola_los_dos_umbrales():
    """Da igual cuál sea más estricto: la banda de vigilancia queda siempre debajo."""
    holgada = Politica.desde_costo_y_capacidad(umbral_costo=2.0, umbral_capacidad=9.0)
    assert (holgada.umbral_abstencion, holgada.umbral_inspeccion) == (2.0, 9.0)
    sobrada = Politica.desde_costo_y_capacidad(umbral_costo=9.0, umbral_capacidad=2.0)
    assert (sobrada.umbral_abstencion, sobrada.umbral_inspeccion) == (2.0, 9.0)


def test_un_ciclo_sin_regimen_estacionario_siempre_se_abstiene():
    """El detector se entrenó solo con ciclos estables: fuera de ahí no sabe qué mira."""
    politica = Politica(umbral_inspeccion=5.0, umbral_abstencion=1.0)
    puntuacion = np.array([0.5, 9.0, 9.0])
    estable = np.array([0, 0, 1])
    assert list(politica.decidir(puntuacion, estable)) == [
        "normal", "inspeccionar", "abstencion"]


def test_las_tres_zonas_suman_uno_en_cada_nivel():
    tabla = pd.DataFrame({
        "objetivo": ["valvula"] * 4,
        "valvula": [100, 100, 90, 90],
        "severidad": ["optimo", "optimo", "retardo pequeno", "retardo pequeno"],
    })
    politica = Politica(umbral_inspeccion=5.0, umbral_abstencion=1.0)
    decision = politica.decidir(np.array([0.5, 2.0, 2.0, 9.0]))
    resumen = resumen_de_politica(tabla, decision)
    assert np.allclose(resumen[["normal", "abstencion", "inspeccionar"]].sum(axis=1), 1.0)


def test_la_auditoria_separa_por_regimen():
    tabla = pd.DataFrame({
        "objetivo": ["acumulador"] * 4,
        "anomalia": [1, 1, 1, 0],
        "refrigerador": [3, 3, 100, 100],
    })
    auditoria = auditoria_por_regimen(tabla, np.array([9.0, 9.0, 0.1, 0.1]),
                                      umbral=1.0, regimen="refrigerador")
    por_estado = auditoria.set_index("refrigerador")
    assert por_estado.loc[3, "recall"] == 1.0
    assert por_estado.loc[100, "recall"] == 0.0
    assert por_estado.loc[100, "falsas_alarmas"] == 0.0
