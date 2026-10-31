"""Repite la comparación con varias particiones para saber si la diferencia es real.

Una sola partición no distingue «Mahalanobis gana» de «Mahalanobis ganó esta vez». Este
script repite el protocolo completo con N semillas distintas sobre los tres objetivos y
publica la media y la desviación de cada métrica, además de **cuántas veces gana cada
alternativa**, que es la pregunta que importa.

Nada más cambia entre repeticiones: misma extracción, mismo filtro de estabilidad, misma
regla de umbral y los mismos tres detectores. Lo único que varía es qué ciclos caen en
entrenamiento y cuáles en prueba.

`por_semilla.csv` incluye también el coste de cada alternativa —`ajuste_s`, `tamano_kb` y
`parametros`— medido con la misma convención que `scripts/run_models.py`. Los tiempos son
medidas físicas de una máquina compartida: varían entre ejecuciones y se leen como órdenes
de magnitud, no como cifras exactas. Ninguna métrica del informe depende de ellos.

Uso:
    uv run python scripts/run_seeds.py --semillas 10
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from inf8239_tf.baseline import AlarmaPorAguja
from inf8239_tf.data import (OBJETIVOS, RAIZ, columnas_aguja, columnas_sensor, etiquetar,
                             particion, solo_normales)
from inf8239_tf.evaluacion import (deteccion_por_severidad, evaluar, linea_de_azar,
                                   umbral_por_falsas_alarmas)
from inf8239_tf.modelos import DetectorVAE, Mahalanobis

CICLOS = RAIZ / "data" / "ciclos.csv"
SALIDA = RAIZ / "reports" / "robustez"
TASA_FALSAS_ALARMAS = 0.05
CAPACIDAD_TURNO = 20
ALTERNATIVAS = ("baseline_agujas", "mahalanobis", "vae")


def una_repeticion(tabla: pd.DataFrame, objetivo: str, semilla: int) -> tuple[list, list]:
    entrena, prueba = particion(tabla, semilla=semilla, objetivo=objetivo)
    normales = solo_normales(entrena)
    corte = int(len(normales) * 2 / 3)
    ajuste, validacion = normales.iloc[:corte], normales.iloc[corte:]
    variables = columnas_sensor(tabla)

    agujas = columnas_aguja(tabla)

    # El reloj cubre **solo el ajuste**, y el tamaño sale del número de parámetros por el
    # ancho de su tipo: float64 en la covarianza de Mahalanobis, float32 en los pesos del
    # VAE. Es la misma convención que `scripts/run_models.py`, para que `ajuste_s` y
    # `tamano_kb` signifiquen lo mismo en `robustez/` y en `modelos_*/`.
    inicio = time.perf_counter()
    alarma = AlarmaPorAguja.ajustar(ajuste, agujas)
    detectores = [("baseline_agujas", alarma, time.perf_counter() - inicio,
                   0.0, len(agujas) * 3)]

    inicio = time.perf_counter()
    maha = Mahalanobis(variables).ajustar(ajuste)
    detectores.append(("mahalanobis", maha, time.perf_counter() - inicio,
                       maha.parametros * 8 / 1024, maha.parametros))

    inicio = time.perf_counter()
    vae = DetectorVAE(variables).ajustar(ajuste, validacion)
    detectores.append(("vae", vae, time.perf_counter() - inicio,
                       vae.parametros * 4 / 1024, vae.parametros))

    metricas, severidades = [], []
    for nombre, detector, segundos, kb, parametros in detectores:
        umbral = umbral_por_falsas_alarmas(detector.puntuar(validacion), TASA_FALSAS_ALARMAS)
        puntuacion = detector.puntuar(prueba)
        fila = evaluar(nombre, prueba, puntuacion, umbral, CAPACIDAD_TURNO, segundos, kb)
        fila.update({"parametros": parametros, "objetivo": objetivo, "semilla": semilla})
        metricas.append(fila)

        for sev in deteccion_por_severidad(prueba, puntuacion, umbral).to_dict("records"):
            sev.update({"objetivo": objetivo, "semilla": semilla, "alternativa": nombre})
            severidades.append(sev)
    return metricas, severidades


def resumir(metricas: pd.DataFrame) -> pd.DataFrame:
    """Media, desviación y el intervalo de confianza del 95 % de cada métrica."""
    filas = []
    for (objetivo, alternativa), grupo in metricas.groupby(["objetivo", "alternativa"]):
        fila = {"objetivo": objetivo, "alternativa": alternativa, "semillas": len(grupo)}
        for metrica in ("pr_auc", "roc_auc", "recall", "falsas_alarmas"):
            valores = grupo[metrica].to_numpy()
            media, desv = valores.mean(), valores.std(ddof=1)
            margen = 1.96 * desv / np.sqrt(len(valores))
            fila[f"{metrica}_media"] = round(float(media), 4)
            fila[f"{metrica}_desv"] = round(float(desv), 4)
            fila[f"{metrica}_ic95"] = f"[{media - margen:.4f}, {media + margen:.4f}]"
        filas.append(fila)
    return pd.DataFrame(filas)


def victorias(metricas: pd.DataFrame, metrica: str = "pr_auc") -> pd.DataFrame:
    """Cuántas veces gana cada alternativa. Es lo que convierte una cifra en un hallazgo."""
    filas = []
    for objetivo, grupo in metricas.groupby("objetivo"):
        conteo = (grupo.loc[grupo.groupby("semilla")[metrica].idxmax(), "alternativa"]
                  .value_counts())
        fila = {"objetivo": objetivo, "semillas": grupo.semilla.nunique()}
        for alternativa in ALTERNATIVAS:
            fila[alternativa] = int(conteo.get(alternativa, 0))
        filas.append(fila)
    return pd.DataFrame(filas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semillas", type=int, default=10)
    args = parser.parse_args()
    semillas = list(range(42, 42 + args.semillas))

    base = pd.read_csv(CICLOS)
    todas_metricas, todas_severidades = [], []
    for objetivo in OBJETIVOS:
        tabla = etiquetar(base, objetivo)
        print(f"\n{objetivo} · {OBJETIVOS[objetivo]['descripcion']}")
        for semilla in semillas:
            metricas, severidades = una_repeticion(tabla, objetivo, semilla)
            todas_metricas.extend(metricas)
            todas_severidades.extend(severidades)
            mejor = max(metricas, key=lambda m: m["pr_auc"])
            print(f"  semilla {semilla}: gana {mejor['alternativa']:<15} "
                  f"PR-AUC {mejor['pr_auc']:.4f}")

    metricas = pd.DataFrame(todas_metricas)
    severidades = pd.DataFrame(todas_severidades)
    SALIDA.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(SALIDA / "por_semilla.csv", index=False)
    severidades.to_csv(SALIDA / "severidad_por_semilla.csv", index=False)

    resumen = resumir(metricas)
    resumen.to_csv(SALIDA / "resumen.csv", index=False)
    ganadores = victorias(metricas)
    ganadores.to_csv(SALIDA / "victorias.csv", index=False)

    print("\n\nMedia e intervalo de confianza del 95 % sobre "
          f"{len(semillas)} particiones:\n")
    print(resumen[["objetivo", "alternativa", "pr_auc_media", "pr_auc_ic95",
                   "roc_auc_media", "recall_media"]].to_string(index=False))
    print("\n\nCuántas veces gana cada alternativa (por PR-AUC):\n")
    print(ganadores.to_string(index=False))

    sev = (severidades[severidades.alternativa.isin(("mahalanobis", "vae"))]
           .groupby(["objetivo", "severidad", "alternativa"]).tasa
           .agg(["mean", "std"]).round(4).reset_index())
    sev.to_csv(SALIDA / "severidad_resumen.csv", index=False)

    (SALIDA / "protocolo.json").write_text(json.dumps({
        "semillas": semillas,
        "objetivos": list(OBJETIVOS),
        "tasa_falsas_alarmas": TASA_FALSAS_ALARMAS,
        "que_cambia_entre_repeticiones": "solo la particion estratificada",
        "linea_de_azar": {o: round(linea_de_azar(etiquetar(base, o).anomalia.to_numpy()), 4)
                          for o in OBJETIVOS},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
