"""Construye la tabla de una fila por ciclo a partir de los 17 archivos de sensor.

Lee data/crudo/, resume cada ciclo en cinco estadísticos por sensor y guarda
data/ciclos.csv junto con un resumen en reports/datos/resumen.json.

Uso:
    uv run python scripts/build_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

from inf8239_tf.data import (AGREGADOS, RAIZ, SEGMENTOS_LENTOS, SEGMENTOS_RAPIDOS, SENSORES,
                             SENSORES_LENTOS, SENSORES_RAPIDOS, SENSORES_VIRTUALES,
                             VALVULA_NIVELES,
                             columnas_aguja, columnas_sensor, construir_tabla, particion,
                             solo_estables, solo_normales, validar_contrato)

SALIDA = RAIZ / "data" / "ciclos.csv"
REPORTES = RAIZ / "reports" / "datos"


def main() -> None:
    tabla = construir_tabla()
    validar_contrato(tabla)
    print(f"Tabla completa: {tabla.shape[0]} ciclos x {len(columnas_sensor(tabla))} "
          f"columnas de sensor")

    estables = solo_estables(tabla)
    descartados = len(tabla) - len(estables)
    print(f"Descartados {descartados} ciclos sin régimen estacionario; quedan {len(estables)}")

    validar_contrato(estables)
    entrena, prueba = particion(estables)
    normales = solo_normales(entrena)

    print(f"\nPartición estratificada 70/30, semilla 42:")
    print(f"  entrenamiento {len(entrena):4d}  de los cuales normales {len(normales):4d}")
    print(f"  prueba        {len(prueba):4d}")
    print(f"\nNiveles de válvula en cada partición:")
    for nivel, etiqueta in VALVULA_NIVELES.items():
        print(f"  {nivel:>3} % {etiqueta:<22} entrenamiento {(entrena.valvula == nivel).sum():4d}"
              f"   prueba {(prueba.valvula == nivel).sum():4d}")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    estables.to_csv(SALIDA, index=False)
    REPORTES.mkdir(parents=True, exist_ok=True)
    (REPORTES / "resumen.json").write_text(json.dumps({
        "ciclos_totales": int(len(tabla)),
        "ciclos_estables": int(len(estables)),
        "descartados_por_transitorio": int(descartados),
        "sensores": len(SENSORES),
        "segmentos_rapidos": SEGMENTOS_RAPIDOS,
        "segmentos_lentos": SEGMENTOS_LENTOS,
        "sensores_rapidos": list(SENSORES_RAPIDOS),
        "sensores_lentos": list(SENSORES_LENTOS),
        "sensores_virtuales": list(SENSORES_VIRTUALES),
        "columnas_sensor": len(columnas_sensor(estables)),
        "columnas_aguja": len(columnas_aguja(estables)),
        "objetivo": "valvula",
        "normal": "valvula == 100",
        "particion": {"tipo": "estratificada por nivel de valvula", "semilla": 42,
                      "entrenamiento": int(len(entrena)), "prueba": int(len(prueba)),
                      "normales_entrenamiento": int(len(normales))},
        "niveles": {str(k): {"etiqueta": v, "ciclos": int((estables.valvula == k).sum())}
                    for k, v in VALVULA_NIVELES.items()},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nGuardado {SALIDA.relative_to(RAIZ)} y {(REPORTES / 'resumen.json').relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
