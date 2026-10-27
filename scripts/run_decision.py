"""Umbral por costo, política de abstención, auditoría por régimen y Green AI.

Toma el detector ganador y lo convierte en una decisión operativa con sus límites.

Uso:
    uv run python scripts/run_decision.py --objetivo valvula --razon-costo 10
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from inf8239_tf.data import (OBJETIVOS, RAIZ, columnas_aguja, columnas_sensor, etiquetar,
                             particion, solo_normales)
from inf8239_tf.baseline import AlarmaPorAguja
from inf8239_tf.decision import (Costos, Politica, auditoria_por_regimen,
                                 resumen_de_politica, sensibilidad_al_costo,
                                 umbral_por_capacidad, umbral_por_costo)
from inf8239_tf.evaluacion import deteccion_por_severidad, evaluar
from inf8239_tf.modelos import DetectorVAE, Mahalanobis

CICLOS = RAIZ / "data" / "ciclos.csv"
RAZONES = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
CAPACIDAD = 0.15   # el equipo puede inspeccionar el 15 % de los ciclos de un turno


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objetivo", choices=sorted(OBJETIVOS), default="valvula")
    parser.add_argument("--razon-costo", type=float, default=10.0,
                        help="cuántas inspecciones inútiles compensa evitar un fallo")
    parser.add_argument("--capacidad", type=float, default=CAPACIDAD,
                        help="fracción de ciclos que el equipo puede inspeccionar")
    args = parser.parse_args()

    salida = RAIZ / "reports" / f"decision_{args.objetivo}"
    salida.mkdir(parents=True, exist_ok=True)

    tabla = etiquetar(pd.read_csv(CICLOS), args.objetivo)
    entrena, prueba = particion(tabla, objetivo=args.objetivo)
    normales = solo_normales(entrena)
    corte = int(len(normales) * 2 / 3)
    ajuste, validacion_normal = normales.iloc[:corte], normales.iloc[corte:]

    # Para fijar el umbral por costo hace falta ver anomalías: se usa el resto del
    # entrenamiento, nunca la prueba.
    validacion = pd.concat([validacion_normal, entrena[entrena.anomalia == 1]])
    variables = columnas_sensor(tabla)

    detectores = {
        "baseline_agujas": AlarmaPorAguja.ajustar(ajuste, columnas_aguja(tabla)),
        "mahalanobis": Mahalanobis(variables).ajustar(ajuste),
        "vae": DetectorVAE(variables).ajustar(ajuste, validacion_normal),
    }

    print(f"Objetivo: {args.objetivo} · {OBJETIVOS[args.objetivo]['descripcion']}")
    print(f"Validación para el umbral: {len(validacion)} ciclos "
          f"({len(validacion_normal)} normales) · prueba: {len(prueba)}\n")

    # --- 1. Sensibilidad del umbral al costo --------------------------------------
    maha = detectores["mahalanobis"]
    punt_val = maha.puntuar(validacion)
    y_val = validacion.anomalia.to_numpy()
    sensibilidad = sensibilidad_al_costo(y_val, punt_val, RAZONES)
    sensibilidad.to_csv(salida / "sensibilidad_costo.csv", index=False)
    print("Umbral óptimo según cuánto más caro sea no detectar un fallo (Mahalanobis):")
    print(sensibilidad.to_string(index=False))

    costos = Costos(fallo_no_detectado=args.razon_costo)
    umbral = umbral_por_costo(y_val, punt_val, costos)
    print(f"\nUmbral elegido con razón {args.razon_costo:g}:1 = {umbral:.4f}\n")

    # --- 2. Política con abstención ------------------------------------------------
    punt_prueba = maha.puntuar(prueba)
    umbral_capacidad = umbral_por_capacidad(punt_prueba, args.capacidad)
    politica = Politica.desde_costo_y_capacidad(umbral, umbral_capacidad)
    decision = politica.decidir(punt_prueba, prueba.estable.to_numpy())
    resumen = resumen_de_politica(prueba, decision)
    resumen.to_csv(salida / "politica.csv", index=False)
    print(f"Umbral por costo {umbral:.3f} · por capacidad del {args.capacidad:.0%} "
          f"{umbral_capacidad:.3f}")
    print(f"Política: inspeccionar por encima de {politica.umbral_inspeccion:.3f}, "
          f"vigilar entre {politica.umbral_abstencion:.3f} y "
          f"{politica.umbral_inspeccion:.3f}")
    print(resumen.to_string(index=False))

    # --- 3. Auditoría por régimen de operación -------------------------------------
    print("\nRendimiento según el estado de OTROS componentes "
          "(¿sirve cuando la máquina no está sana?):")
    auditorias = {}
    otros = [c for c in ("refrigerador", "fuga_bomba", "valvula", "acumulador")
             if c != args.objetivo]
    for regimen in otros:
        auditoria = auditoria_por_regimen(prueba, punt_prueba,
                                          politica.umbral_inspeccion, regimen)
        auditorias[regimen] = auditoria
        auditoria.to_csv(salida / f"regimen_{regimen}.csv", index=False)
        print(f"\n  según {regimen}:")
        print(auditoria.to_string(index=False))

    # --- 4. Green AI ----------------------------------------------------------------
    print("\nGreen AI: costo de cada alternativa")
    print(f"Todos con la MISMA capacidad de inspección, el {args.capacidad:.0%} de los "
          f"ciclos: a igual presupuesto de revisión, ¿quién encuentra más fallos reales?")
    filas = []
    for nombre, detector in detectores.items():
        inicio = time.perf_counter()
        for _ in range(10):
            detector.puntuar(prueba)
        latencia = (time.perf_counter() - inicio) * 1000 / (10 * len(prueba))
        parametros = getattr(detector, "parametros", len(columnas_aguja(tabla)) * 3)
        punt = detector.puntuar(prueba)
        umbral_propio = umbral_por_capacidad(punt, args.capacidad)
        metricas = evaluar(nombre, prueba, punt, umbral_propio, 20, 0.0, 0.0)
        filas.append({"alternativa": nombre, "parametros": parametros,
                      "latencia_ms_por_ciclo": round(latencia, 5),
                      "umbral": round(umbral_propio, 4),
                      "pr_auc": metricas["pr_auc"],
                      "precision_de_lo_inspeccionado": round(
                          1 - metricas["falsas_alarmas"] * (prueba.anomalia == 0).sum()
                          / max(metricas["marcados"], 1), 4),
                      "recall": metricas["recall"],
                      "falsas_alarmas": metricas["falsas_alarmas"]})
    verde = pd.DataFrame(filas)
    verde.to_csv(salida / "green_ai.csv", index=False)
    print(verde.to_string(index=False))

    (salida / "protocolo.json").write_text(json.dumps({
        "objetivo": args.objetivo,
        "razon_costo_fn_fa": args.razon_costo,
        "umbral_costo": round(umbral, 4),
        "umbral_capacidad": round(umbral_capacidad, 4),
        "capacidad": args.capacidad,
        "umbral_inspeccion": round(politica.umbral_inspeccion, 4),
        "umbral_abstencion": round(politica.umbral_abstencion, 4),
        "validacion_total": len(validacion),
        "validacion_normales": len(validacion_normal),
        "regla_abstencion_extra": "todo ciclo sin regimen estacionario se abstiene",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
