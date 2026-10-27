"""El baseline: la alarma por umbral de cada aguja, que es lo que hace la planta hoy.

Un panel de control muestra una aguja por sensor con una banda verde y una zona roja. El
operador vigila **cada aguja contra su propio límite** y no puede ver si la combinación de
lecturas es imposible. Reproducir eso con fidelidad es lo que hace que la comparación sea
honesta: si el modelo gana, gana contra la práctica real y no contra un hombre de paja.

Decisiones declaradas antes de medir:

- Se usa **la media del ciclo de cada sensor**, que es la lectura que un operador vería.
- Los límites se calculan **solo con los ciclos normales de entrenamiento**, con los
  cuantiles 0.5 % y 99.5 %. No se ajustan después mirando el resultado.
- La puntuación continua es la **desviación máxima entre todas las agujas**, normalizada
  para que valga 1 justo en el límite. Así la misma regla da una alarma binaria —la zona
  roja— y una puntuación comparable por PR-AUC con los modelos.
- **No se codifica ninguna regla física conocida.** En una planta real nadie conoce la
  regla que genera el fallo; si la conociera, no necesitaría el modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CUANTIL_INFERIOR = 0.005
CUANTIL_SUPERIOR = 0.995


@dataclass
class AlarmaPorAguja:
    """Límite inferior, superior y centro de cada aguja, aprendidos de lo normal."""

    inferior: pd.Series
    superior: pd.Series
    centro: pd.Series
    columnas: list[str]

    @classmethod
    def ajustar(cls, normales: pd.DataFrame, columnas: list[str]) -> "AlarmaPorAguja":
        datos = normales[columnas]
        return cls(
            inferior=datos.quantile(CUANTIL_INFERIOR),
            superior=datos.quantile(CUANTIL_SUPERIOR),
            centro=datos.median(),
            columnas=list(columnas),
        )

    def desviaciones(self, tabla: pd.DataFrame) -> pd.DataFrame:
        """Cuánto se sale cada aguja de su banda, en unidades de media banda.

        Vale 0 en el centro de la banda normal y 1 justo en el límite, de modo que
        cualquier valor por encima de 1 es zona roja.
        """
        datos = tabla[self.columnas]
        arriba = (datos - self.centro) / (self.superior - self.centro).replace(0, np.nan)
        abajo = (self.centro - datos) / (self.centro - self.inferior).replace(0, np.nan)
        return pd.DataFrame(np.maximum(arriba, abajo), index=tabla.index,
                            columns=self.columnas).fillna(0.0)

    def puntuar(self, tabla: pd.DataFrame) -> np.ndarray:
        """Puntuación de anomalía: la aguja que más se sale."""
        return self.desviaciones(tabla).max(axis=1).to_numpy()

    def alarma(self, tabla: pd.DataFrame) -> np.ndarray:
        """La alarma binaria de la planta: alguna aguja entró en zona roja."""
        return (self.puntuar(tabla) > 1.0).astype(int)

    def aguja_culpable(self, tabla: pd.DataFrame) -> pd.Series:
        """Qué sensor disparó la alarma. Es la explicación que el operador necesita."""
        return self.desviaciones(tabla).idxmax(axis=1)
