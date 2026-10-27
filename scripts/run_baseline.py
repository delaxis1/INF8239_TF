"""Ejecuta el baseline de alarmas por aguja y deja la marca que los modelos deben batir.

Uso:
    uv run python scripts/run_baseline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from inf8239_tf.baseline import AlarmaPorAguja
from inf8239_tf.data import RAIZ, columnas_aguja, particion, solo_normales
from inf8239_tf.evaluacion import (deteccion_por_severidad, evaluar, linea_de_azar,
                                   umbral_por_falsas_alarmas)

CICLOS = RAIZ / "data" / "ciclos.csv"
SALIDA = RAIZ / "reports" / "baseline"
TASA_FALSAS_ALARMAS = 0.05
CAPACIDAD_TURNO = 20


def main() -> None:
    tabla = pd.read_csv(CICLOS)
    entrena, prueba = particion(tabla)
    normales = solo_normales(entrena)
    agujas = columnas_aguja(tabla)

    # Validación: un tercio de los normales de entrenamiento, solo para fijar el umbral.
    corte = int(len(normales) * 2 / 3)
    ajuste, validacion = normales.iloc[:corte], normales.iloc[corte:]

    inicio = time.perf_counter()
    alarma = AlarmaPorAguja.ajustar(ajuste, agujas)
    segundos = time.perf_counter() - inicio

    umbral_rojo = 1.0
    umbral_tasa = umbral_por_falsas_alarmas(alarma.puntuar(validacion), TASA_FALSAS_ALARMAS)
    puntuacion = alarma.puntuar(prueba)

    print(f"Agujas vigiladas: {len(agujas)}")
    print(f"Normales para ajustar el límite: {len(ajuste)} · para fijar el umbral: "
          f"{len(validacion)}")
    print(f"Prueba: {len(prueba)} ciclos · prevalencia de anomalías "
          f"{linea_de_azar(prueba.anomalia.to_numpy()):.3f}\n")

    filas = []
    for nombre, umbral in [("alarma_zona_roja", umbral_rojo),
                           (f"alarma_al_{int(TASA_FALSAS_ALARMAS * 100)}pct_falsas", umbral_tasa)]:
        metricas = evaluar(nombre, prueba, puntuacion, umbral, CAPACIDAD_TURNO, segundos, 0.0)
        filas.append(metricas)
        print(f"--- {nombre} (umbral {umbral:.3f}) ---")
        for clave, valor in metricas.items():
            if clave != "alternativa":
                print(f"  {clave:<22} {valor}")
        severidad = deteccion_por_severidad(prueba, puntuacion, umbral)
        print(severidad.to_string(index=False))
        print()
        SALIDA.mkdir(parents=True, exist_ok=True)
        severidad.to_csv(SALIDA / f"severidad_{nombre}.csv", index=False)

    pd.DataFrame(filas).to_csv(SALIDA / "metricas.csv", index=False)

    culpables = alarma.aguja_culpable(prueba[prueba.anomalia == 1]).value_counts()
    print("Aguja que más se sale en los ciclos con válvula degradada:")
    print(culpables.head(5).to_string())

    (SALIDA / "configuracion.json").write_text(json.dumps({
        "agujas": agujas,
        "cuantiles": [0.005, 0.995],
        "tasa_falsas_alarmas_objetivo": TASA_FALSAS_ALARMAS,
        "capacidad_por_turno": CAPACIDAD_TURNO,
        "normales_ajuste": len(ajuste),
        "normales_validacion": len(validacion),
        "umbral_zona_roja": umbral_rojo,
        "umbral_por_tasa": round(float(umbral_tasa), 4),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
