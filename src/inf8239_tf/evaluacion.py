"""Métricas de detección de anomalías, con el umbral y el costo que exige el tema.

Por qué estas métricas y no otras:

- **PR-AUC** y no exactitud ni ROC-AUC. La exactitud premia al que dice «todo normal». El
  ROC-AUC se infla cuando las clases están desbalanceadas, en cualquiera de los dos
  sentidos. Con PR-AUC hay que declarar siempre la **línea de azar**, que es la proporción
  de anomalías: un PR-AUC de 0.80 con 75 % de anomalías es casi no hacer nada.
- **Precisión en los k primeros**, porque el analista no revisa todo: revisa los k casos
  que le caben en el turno. Es la métrica que mide la utilidad real del sistema.
- **Detección por nivel de severidad**, que es el corazón del trabajo: ¿el sistema avisa
  cuando el retardo es pequeño, o solo cuando ya está cerca del fallo total, que es
  justo cuando el operador también lo vería?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def linea_de_azar(y: np.ndarray) -> float:
    """PR-AUC que obtendría puntuar al azar: la prevalencia de anomalías."""
    return float(np.mean(y))


def precision_en_k(y: np.ndarray, puntuacion: np.ndarray, k: int) -> float:
    """De los k casos peor puntuados, qué fracción eran anomalías de verdad."""
    k = min(k, len(y))
    orden = np.argsort(-puntuacion)[:k]
    return float(np.mean(y[orden]))


def umbral_por_falsas_alarmas(puntuacion_normales: np.ndarray, tasa: float) -> float:
    """Umbral que deja pasar como mucho `tasa` de falsas alarmas sobre ciclos normales.

    Se calcula **solo con normales de validación**, nunca con el conjunto de prueba ni
    con las anomalías. Es la regla preinscrita: se fija la tasa de falsas alarmas que la
    operación tolera y el umbral sale de ahí, no al revés.
    """
    if not 0.0 < tasa < 1.0:
        raise ValueError("La tasa de falsas alarmas debe estar entre 0 y 1")
    return float(np.quantile(puntuacion_normales, 1.0 - tasa))


def deteccion_por_severidad(tabla: pd.DataFrame, puntuacion: np.ndarray,
                            umbral: float) -> pd.DataFrame:
    """Qué fracción de cada nivel de degradación supera el umbral.

    Es la tabla que responde la pregunta del trabajo: dónde empieza a ver el sistema.
    """
    objetivo = tabla.objetivo.iloc[0] if "objetivo" in tabla.columns else "valvula"
    filas = []
    for nivel in sorted(tabla[objetivo].unique(), reverse=True):
        mascara = (tabla[objetivo] == nivel).to_numpy()
        detectados = int((puntuacion[mascara] > umbral).sum())
        filas.append({
            "nivel": int(nivel),
            "severidad": tabla.loc[mascara, "severidad"].iloc[0],
            "ciclos": int(mascara.sum()),
            "detectados": detectados,
            "tasa": round(detectados / int(mascara.sum()), 4),
        })
    return pd.DataFrame(filas)


def evaluar(nombre: str, tabla: pd.DataFrame, puntuacion: np.ndarray, umbral: float,
            k: int, segundos_ajuste: float, kb: float) -> dict:
    """Todas las métricas de una alternativa, listas para la tabla comparativa."""
    y = tabla.anomalia.to_numpy()
    marcados = puntuacion > umbral
    verdaderos = int((marcados & (y == 1)).sum())
    falsos = int((marcados & (y == 0)).sum())
    positivos = int(y.sum())
    negativos = int((y == 0).sum())
    return {
        "alternativa": nombre,
        "pr_auc": round(float(average_precision_score(y, puntuacion)), 4),
        "linea_de_azar": round(linea_de_azar(y), 4),
        "roc_auc": round(float(roc_auc_score(y, puntuacion)), 4),
        f"precision_en_{k}": round(precision_en_k(y, puntuacion, k), 4),
        "recall": round(verdaderos / positivos, 4) if positivos else 0.0,
        "falsas_alarmas": round(falsos / negativos, 4) if negativos else 0.0,
        "marcados": int(marcados.sum()),
        "umbral": round(float(umbral), 4),
        "ajuste_s": round(segundos_ajuste, 4),
        "tamano_kb": round(kb, 2),
    }


def curva(y: np.ndarray, puntuacion: np.ndarray) -> pd.DataFrame:
    precision, recall, umbrales = precision_recall_curve(y, puntuacion)
    return pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1],
                         "umbral": umbrales})
