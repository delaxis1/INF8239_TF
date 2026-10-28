"""Un caso trabajado y las dos figuras del informe.

Tres salidas, todas derivadas de los mismos artefactos que el resto del trabajo:

1. **El caso.** Un ciclo degradado que el detector marca y en el que **ninguna aguja sale
   de su banda**. Es la tesis del trabajo hecha visible en un solo ejemplo: el operador no
   habría visto nada.
2. **Figura de detección por severidad.** Las tres alternativas sobre los cuatro niveles de
   degradación de la válvula.
3. **Figura de la representación.** ROC-AUC frente al número de tramos temporales, que es
   el hallazgo principal.

El caso se elige por una regla declarada **antes** de mirarlo: entre los ciclos con el
nivel de degradación más leve —el que el operador no ve— que el modelo marca y que no
disparan ninguna aguja, el de **aguja más tranquila**. Es decir, el caso en el que el panel
estaba más lejos de sospechar.

Uso:
    uv run python scripts/run_caso_y_figuras.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt  # noqa: E402

from inf8239_tf.baseline import AlarmaPorAguja  # noqa: E402
from inf8239_tf.data import (CRUDO, RAIZ, SENSORES_LENTOS, SENSORES_RAPIDOS,  # noqa: E402
                             columnas_aguja, columnas_sensor, etiquetar, particion,
                             solo_normales)
from inf8239_tf.evaluacion import (deteccion_por_severidad,  # noqa: E402
                                   umbral_por_falsas_alarmas)
from inf8239_tf.modelos import Mahalanobis  # noqa: E402

CICLOS = RAIZ / "data" / "ciclos.csv"
CASO = RAIZ / "reports" / "caso"
FIGURAS = RAIZ / "reports" / "figuras"
TASA_FALSAS_ALARMAS = 0.05

NAVY, ROJO, GRIS = "#12355B", "#9B2C2C", "#5F5E5A"


def preparar():
    tabla = etiquetar(pd.read_csv(CICLOS), "valvula")
    entrena, prueba = particion(tabla, objetivo="valvula")
    normales = solo_normales(entrena)
    corte = int(len(normales) * 2 / 3)
    ajuste, validacion = normales.iloc[:corte], normales.iloc[corte:]
    agujas = AlarmaPorAguja.ajustar(ajuste, columnas_aguja(tabla))
    maha = Mahalanobis(columnas_sensor(tabla)).ajustar(ajuste)
    umbral = umbral_por_falsas_alarmas(maha.puntuar(validacion), TASA_FALSAS_ALARMAS)
    return tabla, prueba, agujas, maha, umbral


def el_caso(prueba, agujas, maha, umbral) -> dict:
    puntuacion = maha.puntuar(prueba)
    desviaciones = agujas.desviaciones(prueba)
    peor_aguja = desviaciones.max(axis=1).to_numpy()
    y = prueba.anomalia.to_numpy()

    invisibles = (y == 1) & (puntuacion > umbral) & (peor_aguja <= 1.0)
    candidatos = prueba[invisibles & (prueba.severidad == "retardo pequeno").to_numpy()]
    orden = np.argsort(peor_aguja[invisibles & (prueba.severidad == "retardo pequeno").to_numpy()])
    elegido = candidatos.iloc[[orden[0]]]
    indice = elegido.index[0]

    aportes = maha.contribuciones(elegido).iloc[0].sort_values(ascending=False)
    desv = desviaciones.loc[indice].sort_values(ascending=False)

    detalle = {
        "ciclo": int(elegido.ciclo.iloc[0]),
        "severidad": elegido.severidad.iloc[0],
        "puntuacion": round(float(puntuacion[prueba.index.get_loc(indice)]), 3),
        "umbral": round(float(umbral), 3),
        "peor_aguja": desv.index[0],
        "desviacion_peor_aguja": round(float(desv.iloc[0]), 3),
        "agujas_en_zona_roja": int((desv > 1.0).sum()),
        "variables_que_lo_explican": [
            {"variable": k, "aporte": round(float(v), 2)} for k, v in aportes.head(5).items()
        ],
        "degradados_marcados_sin_ninguna_aguja_en_rojo": int(invisibles.sum()),
        "degradados_totales": int((y == 1).sum()),
        "regla_de_seleccion": ("entre los ciclos con el nivel mas leve que el modelo marca "
                               "y que no disparan ninguna aguja, el de aguja mas tranquila"),
    }
    CASO.mkdir(parents=True, exist_ok=True)
    (CASO / "caso.json").write_text(json.dumps(detalle, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    desv.head(8).rename("desviacion").to_frame().to_csv(CASO / "agujas.csv")
    aportes.head(8).rename("aporte").to_frame().to_csv(CASO / "contribuciones.csv")
    return detalle


def figura_severidad(prueba, agujas, maha, umbral_maha) -> None:
    # Las tasas ya están calculadas y versionadas por scripts/run_models.py.
    sev = pd.read_csv(RAIZ / "reports" / "modelos_valvula" / "severidad_comparada.csv")
    sev = sev.sort_values("nivel", ascending=False)

    fig, eje = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(sev))
    estilos = [("baseline_agujas", "Alarma por aguja", GRIS, "o", "--"),
               ("mahalanobis", "Mahalanobis", NAVY, "s", "-"),
               ("vae", "VAE", ROJO, "^", "-.")]
    for columna, etiqueta, color, marca, linea in estilos:
        eje.plot(x, sev[columna] * 100, marker=marca, linestyle=linea, color=color,
                 label=etiqueta, linewidth=2, markersize=7)
    eje.set_xticks(x)
    eje.set_xticklabels([s.replace(" ", "\n") for s in sev.severidad], fontsize=8)
    eje.set_ylabel("Ciclos detectados (%)")
    eje.set_ylim(-4, 108)
    eje.axvline(0.5, color="#C9CED6", linewidth=1)
    eje.text(0.02, 104, "ciclos sanos:\naquí son falsas alarmas", fontsize=7, color=GRIS,
             va="top")
    eje.legend(frameon=False, fontsize=9, loc="center right")
    eje.spines[["top", "right"]].set_visible(False)
    eje.grid(axis="y", color="#EEF1F5", linewidth=1)
    eje.set_axisbelow(True)
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / "deteccion_por_severidad.png", dpi=200)
    plt.close(fig)


def figura_representacion(tabla) -> None:
    """ROC-AUC frente al número de tramos temporales. El hallazgo principal."""
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    perfil = np.loadtxt(CRUDO / "profile.txt")
    estables = perfil[:, 4] == 0
    y = (perfil[estables, 1] != 100).astype(int)
    generador = np.random.default_rng(42)
    indices = np.arange(estables.sum())
    generador.shuffle(indices)
    corte = int(len(indices) * 0.7)
    entrena, prueba = indices[:corte], indices[corte:]
    normales = entrena[y[entrena] == 0]

    tramos = [1, 2, 3, 6, 12, 24]
    auc = []
    for n in tramos:
        bloques = []
        for sensor in SENSORES_RAPIDOS:
            matriz = np.loadtxt(CRUDO / f"{sensor}.txt")[estables]
            bloques.append(np.column_stack([t.mean(axis=1)
                                            for t in np.array_split(matriz, n, axis=1)]))
        X = np.column_stack(bloques)
        escala = StandardScaler().fit(X[normales])
        covarianza = LedoitWolf().fit(escala.transform(X[normales]))
        distancia = np.sqrt(covarianza.mahalanobis(escala.transform(X[prueba])))
        auc.append(roc_auc_score(y[prueba], distancia))
        print(f"  {n:>2} tramos x {len(SENSORES_RAPIDOS)} sensores = {X.shape[1]:>3} "
              f"variables   ROC-AUC {auc[-1]:.4f}")

    fig, eje = plt.subplots(figsize=(6.6, 3.4))
    eje.plot(tramos, auc, marker="o", color=NAVY, linewidth=2, markersize=7)
    eje.axhline(0.5, color=GRIS, linestyle=":", linewidth=1.2)
    eje.text(24, 0.515, "azar", fontsize=8, color=GRIS, ha="right")
    eje.annotate(f"solo la media\ndel ciclo\n{auc[0]:.2f}", xy=(1, auc[0]),
                 xytext=(2.4, auc[0] + 0.12), fontsize=8, color=ROJO,
                 arrowprops=dict(arrowstyle="->", color=ROJO, linewidth=1.2))
    eje.annotate(f"seis tramos\n{auc[3]:.3f}", xy=(6, auc[3]), xytext=(8, auc[3] - 0.17),
                 fontsize=8, color=NAVY,
                 arrowprops=dict(arrowstyle="->", color=NAVY, linewidth=1.2))
    eje.set_xlabel("Tramos temporales por sensor")
    eje.set_ylabel("ROC-AUC (Mahalanobis)")
    eje.set_xticks(tramos)
    eje.set_ylim(0.45, 1.04)
    eje.spines[["top", "right"]].set_visible(False)
    eje.grid(axis="y", color="#EEF1F5", linewidth=1)
    eje.set_axisbelow(True)
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / "representacion.png", dpi=200)
    plt.close(fig)
    pd.DataFrame({"tramos": tramos, "roc_auc": np.round(auc, 4)}).to_csv(
        FIGURAS / "representacion.csv", index=False)


def main() -> None:
    tabla, prueba, agujas, maha, umbral = preparar()

    print("=== El caso ===")
    detalle = el_caso(prueba, agujas, maha, umbral)
    print(f"Ciclo {detalle['ciclo']} · {detalle['severidad']}")
    print(f"  puntuación {detalle['puntuacion']} contra un umbral de {detalle['umbral']}")
    print(f"  agujas en zona roja: {detalle['agujas_en_zona_roja']}")
    print(f"  la que más se acerca: {detalle['peor_aguja']} al "
          f"{detalle['desviacion_peor_aguja']:.0%} de su límite")
    print("  lo explican:", ", ".join(v["variable"]
                                      for v in detalle["variables_que_lo_explican"][:3]))
    print(f"\n  {detalle['degradados_marcados_sin_ninguna_aguja_en_rojo']} de "
          f"{detalle['degradados_totales']} ciclos degradados se detectan sin que ninguna "
          f"aguja salga de banda")

    print("\n=== Figura 1: detección por severidad ===")
    figura_severidad(prueba, agujas, maha, umbral)
    print("  guardada")

    print("\n=== Figura 2: la representación ===")
    figura_representacion(tabla)
    print(f"\nGuardado en {CASO.relative_to(RAIZ)} y {FIGURAS.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
