"""Del puntaje a la decisión: umbral por costo, abstención y auditoría por régimen.

Un detector devuelve un número. Convertirlo en «inspeccione este equipo» exige tres cosas
que no son estadísticas sino de operación:

1. **Un umbral justificado por el costo.** Una falsa alarma cuesta una inspección que no
   encuentra nada. Un fallo no detectado cuesta una parada no planificada. La relación
   entre esos dos costos es una decisión de la institución, no del modelo, y el umbral
   óptimo se deduce de ella.
2. **Una zona de abstención.** Entre «claramente normal» y «claramente degradado» hay
   ciclos que el detector no distingue. Decir «no sé» es una respuesta válida y mucho más
   útil que una respuesta segura y equivocada.
3. **Una auditoría por régimen de operación.** Un detector que solo funciona cuando el
   resto de la máquina está sana no sirve en una planta, donde los componentes se degradan
   a la vez. Aquí el papel de los subgrupos lo juegan los estados de los otros
   componentes: refrigerador, bomba y, según el objetivo, válvula o acumulador.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costos:
    """Cuánto cuesta equivocarse en cada dirección, en unidades de inspección.

    `falsa_alarma` se fija en 1 por convención: todo se mide en inspecciones inútiles.
    `fallo_no_detectado` es **la decisión de valor**: cuántas inspecciones inútiles
    compensa evitar una parada no planificada.
    """

    fallo_no_detectado: float
    falsa_alarma: float = 1.0

    def total(self, y: np.ndarray, marcados: np.ndarray) -> float:
        falsos_positivos = int((marcados & (y == 0)).sum())
        falsos_negativos = int((~marcados & (y == 1)).sum())
        return falsos_positivos * self.falsa_alarma + falsos_negativos * self.fallo_no_detectado


def barrido_de_costo(y: np.ndarray, puntuacion: np.ndarray, costos: Costos,
                     n: int = 400) -> pd.DataFrame:
    """Costo total para cada umbral candidato, de más permisivo a más estricto."""
    candidatos = np.quantile(puntuacion, np.linspace(0.0, 1.0, n))
    filas = []
    for umbral in np.unique(candidatos):
        marcados = puntuacion > umbral
        filas.append({
            "umbral": float(umbral),
            "marcados": int(marcados.sum()),
            "falsos_positivos": int((marcados & (y == 0)).sum()),
            "falsos_negativos": int((~marcados & (y == 1)).sum()),
            "costo": costos.total(y, marcados),
        })
    return pd.DataFrame(filas)


def umbral_por_costo(y: np.ndarray, puntuacion: np.ndarray, costos: Costos) -> float:
    """El umbral que minimiza el costo esperado.

    Se calcula sobre **validación**, nunca sobre el conjunto de prueba: el umbral es parte
    del modelo, y ajustarlo con los datos de evaluación sería mirar las respuestas.
    """
    barrido = barrido_de_costo(y, puntuacion, costos)
    return float(barrido.loc[barrido.costo.idxmin(), "umbral"])


def sensibilidad_al_costo(y: np.ndarray, puntuacion: np.ndarray,
                          razones: tuple[float, ...]) -> pd.DataFrame:
    """Cómo se mueve el umbral óptimo según cuánto más caro sea no detectar un fallo.

    Si el umbral apenas cambia en un rango amplio de razones, la decisión es robusta y no
    hace falta conocer el costo con precisión. Si cambia mucho, hay que medirlo antes de
    desplegar, y eso es un resultado del trabajo.
    """
    filas = []
    for razon in razones:
        costos = Costos(fallo_no_detectado=razon)
        barrido = barrido_de_costo(y, puntuacion, costos)
        mejor = barrido.loc[barrido.costo.idxmin()]
        marcados = puntuacion > mejor.umbral
        filas.append({
            "razon_fn_fa": razon,
            "umbral": round(float(mejor.umbral), 4),
            "marcados": int(mejor.marcados),
            "recall": round(float((marcados & (y == 1)).sum() / max(int(y.sum()), 1)), 4),
            "falsas_alarmas": round(float((marcados & (y == 0)).sum()
                                          / max(int((y == 0).sum()), 1)), 4),
            "costo": float(mejor.costo),
        })
    return pd.DataFrame(filas)


def umbral_por_capacidad(puntuacion: np.ndarray, fraccion: float) -> float:
    """Umbral que marca exactamente la fracción de ciclos que el equipo puede inspeccionar.

    El costo y la capacidad son restricciones distintas y suelen estar en conflicto: con
    los fallos 10 veces más caros que una inspección inútil, el óptimo de costo manda
    revisarlo casi todo, cosa que ninguna planta puede hacer.
    """
    if not 0.0 < fraccion <= 1.0:
        raise ValueError("La capacidad debe ser una fracción entre 0 y 1")
    return float(np.quantile(puntuacion, 1.0 - fraccion))


@dataclass(frozen=True)
class Politica:
    """Tres zonas: normal, abstención e inspección.

    La abstención no es un adorno: es **la distancia entre lo que convendría revisar y lo
    que se puede revisar**. Los ciclos cuya puntuación supera el umbral de costo pero no
    entra en la capacidad del turno no se declaran normales ni se inspeccionan: quedan en
    vigilancia, que es la respuesta honesta.

    Cuando la capacidad sobra, la banda se cierra sola y el sistema inspecciona todo lo que
    el costo aconseja.
    """

    umbral_inspeccion: float
    umbral_abstencion: float

    @classmethod
    def desde_costo_y_capacidad(cls, umbral_costo: float, umbral_capacidad: float
                                ) -> "Politica":
        return cls(umbral_inspeccion=max(umbral_costo, umbral_capacidad),
                   umbral_abstencion=min(umbral_costo, umbral_capacidad))

    def __post_init__(self) -> None:
        if self.umbral_abstencion > self.umbral_inspeccion:
            raise ValueError("La banda de abstención debe quedar por debajo de la de "
                             "inspección")

    def decidir(self, puntuacion: np.ndarray, estable: np.ndarray | None = None
                ) -> np.ndarray:
        """Devuelve 'normal', 'abstencion' o 'inspeccionar' para cada ciclo.

        Un ciclo que no alcanzó régimen estacionario **nunca** recibe veredicto: el
        detector se entrenó solo con ciclos estables, así que fuera de ese régimen no sabe
        lo que está mirando. Es la abstención mejor fundada del sistema.
        """
        decision = np.where(puntuacion > self.umbral_inspeccion, "inspeccionar",
                            np.where(puntuacion > self.umbral_abstencion, "abstencion",
                                     "normal"))
        if estable is not None:
            decision = np.where(estable == 0, decision, "abstencion")
        return decision


def resumen_de_politica(tabla: pd.DataFrame, decision: np.ndarray) -> pd.DataFrame:
    """Qué decide el sistema en cada nivel de degradación."""
    objetivo = tabla.objetivo.iloc[0]
    filas = []
    for nivel in sorted(tabla[objetivo].unique(), reverse=True):
        mascara = (tabla[objetivo] == nivel).to_numpy()
        total = int(mascara.sum())
        fila = {"nivel": int(nivel),
                "severidad": tabla.loc[mascara, "severidad"].iloc[0],
                "ciclos": total}
        for estado in ("normal", "abstencion", "inspeccionar"):
            fila[estado] = round(float((decision[mascara] == estado).sum() / total), 4)
        filas.append(fila)
    return pd.DataFrame(filas)


def auditoria_por_regimen(tabla: pd.DataFrame, puntuacion: np.ndarray, umbral: float,
                          regimen: str) -> pd.DataFrame:
    """Recall y falsas alarmas del detector según el estado de OTRO componente.

    Es la pregunta que decide si el sistema sirve en una planta real, donde los
    componentes no se degradan de uno en uno.
    """
    y = tabla.anomalia.to_numpy()
    marcados = puntuacion > umbral
    filas = []
    for valor in sorted(tabla[regimen].unique(), reverse=True):
        m = (tabla[regimen] == valor).to_numpy()
        positivos, negativos = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        filas.append({
            regimen: int(valor),
            "ciclos": int(m.sum()),
            "recall": round(float((marcados[m] & (y[m] == 1)).sum() / positivos), 4)
            if positivos else None,
            "falsas_alarmas": round(float((marcados[m] & (y[m] == 0)).sum() / negativos), 4)
            if negativos else None,
        })
    return pd.DataFrame(filas)
