"""Compara el baseline de agujas con Mahalanobis y el VAE, bajo el mismo protocolo.

Las tres alternativas ven exactamente los mismos ciclos: se ajustan con los 172 normales
de ajuste, fijan su umbral con los 86 normales de validación al 5 % de falsas alarmas, y
se miden una sola vez sobre los 435 ciclos de prueba.

Uso:
    uv run python scripts/run_models.py
    uv run python scripts/run_models.py --sin-virtuales   # ablación CE, CP y SE
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from inf8239_tf.baseline import AlarmaPorAguja
from inf8239_tf.data import RAIZ, columnas_aguja, columnas_sensor, particion, solo_normales
from inf8239_tf.evaluacion import (curva, deteccion_por_severidad, evaluar, linea_de_azar,
                                   umbral_por_falsas_alarmas)
from inf8239_tf.modelos import DetectorVAE, Mahalanobis

CICLOS = RAIZ / "data" / "ciclos.csv"
TASA_FALSAS_ALARMAS = 0.05
CAPACIDAD_TURNO = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sin-virtuales", action="store_true",
                        help="excluye CE, CP y SE, que no se miden sino que se calculan")
    args = parser.parse_args()

    salida = RAIZ / "reports" / ("modelos_sin_virtuales" if args.sin_virtuales else "modelos")
    salida.mkdir(parents=True, exist_ok=True)

    tabla = pd.read_csv(CICLOS)
    entrena, prueba = particion(tabla)
    normales = solo_normales(entrena)
    corte = int(len(normales) * 2 / 3)
    ajuste, validacion = normales.iloc[:corte], normales.iloc[corte:]

    incluir = not args.sin_virtuales
    agujas = columnas_aguja(tabla, incluir_virtuales=incluir)
    variables = columnas_sensor(tabla, incluir_virtuales=incluir)

    print(f"{'Ablación sin sensores virtuales' if args.sin_virtuales else 'Conjunto completo'}: "
          f"{len(agujas)} agujas, {len(variables)} variables")
    print(f"Ajuste {len(ajuste)} · validación {len(validacion)} · prueba {len(prueba)} "
          f"(prevalencia {linea_de_azar(prueba.anomalia.to_numpy()):.3f})\n")

    resultados, severidades = [], {}

    inicio = time.perf_counter()
    alarma = AlarmaPorAguja.ajustar(ajuste, agujas)
    segundos = time.perf_counter() - inicio
    detectores = [("baseline_agujas", alarma, segundos, 0.0, len(agujas) * 3)]

    inicio = time.perf_counter()
    maha = Mahalanobis(variables).ajustar(ajuste)
    segundos = time.perf_counter() - inicio
    detectores.append(("mahalanobis", maha, segundos, maha.parametros * 8 / 1024,
                       maha.parametros))

    inicio = time.perf_counter()
    vae = DetectorVAE(variables).ajustar(ajuste, validacion)
    segundos = time.perf_counter() - inicio
    detectores.append(("vae", vae, segundos, vae.parametros * 4 / 1024, vae.parametros))
    print(f"VAE: parada temprana en la época {vae.mejor_epoca} de {len(vae.historial)} "
          f"ejecutadas · {vae.parametros} parámetros\n")

    for nombre, detector, segundos, kb, parametros in detectores:
        umbral = umbral_por_falsas_alarmas(detector.puntuar(validacion), TASA_FALSAS_ALARMAS)
        puntuacion = detector.puntuar(prueba)
        metricas = evaluar(nombre, prueba, puntuacion, umbral, CAPACIDAD_TURNO, segundos, kb)
        metricas["parametros"] = parametros
        resultados.append(metricas)

        severidad = deteccion_por_severidad(prueba, puntuacion, umbral)
        severidades[nombre] = severidad
        severidad.to_csv(salida / f"severidad_{nombre}.csv", index=False)
        curva(prueba.anomalia.to_numpy(), puntuacion).to_csv(
            salida / f"curva_pr_{nombre}.csv", index=False)

    comparacion = pd.DataFrame(resultados)
    comparacion.to_csv(salida / "comparacion.csv", index=False)
    print(comparacion.to_string(index=False))

    print("\nDetección por nivel de degradación, al 5 % de falsas alarmas:")
    unida = severidades["baseline_agujas"][["valvula_%", "severidad", "ciclos"]].copy()
    for nombre, tabla_sev in severidades.items():
        unida[nombre] = tabla_sev["tasa"].to_numpy()
    print(unida.to_string(index=False))
    unida.to_csv(salida / "severidad_comparada.csv", index=False)

    pd.DataFrame(vae.historial).to_csv(salida / "historial_vae.csv", index=False)
    (salida / "protocolo.json").write_text(json.dumps({
        "sensores_virtuales_incluidos": incluir,
        "variables": len(variables),
        "agujas": len(agujas),
        "ajuste": len(ajuste), "validacion": len(validacion), "prueba": len(prueba),
        "tasa_falsas_alarmas_objetivo": TASA_FALSAS_ALARMAS,
        "capacidad_por_turno": CAPACIDAD_TURNO,
        "vae": {"latente": vae.latente, "oculta": vae.oculta, "beta": vae.beta,
                "mejor_epoca": vae.mejor_epoca, "epocas_ejecutadas": len(vae.historial),
                "parametros": vae.parametros},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
