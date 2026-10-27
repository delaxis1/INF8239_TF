"""Los dos candidatos: distancia de Mahalanobis y VAE.

Los dos se entrenan **solo con ciclos normales** y puntúan cuánto se aleja un ciclo de lo
aprendido. La diferencia está en qué entienden por «lo normal»:

- **Mahalanobis** modela una sola nube gaussiana y sus correlaciones lineales. Es el
  competidor honesto del VAE: si la relación entre sensores es lineal, esto basta, no hace
  falta ninguna red, y decirlo sería el resultado del trabajo. Con 85 variables y 172
  ciclos la covarianza empírica es singular, así que se usa el encogimiento de
  Ledoit-Wolf, que es la forma estándar de estimarla cuando hay más variables que muestras.
- **VAE** aprende una superficie que puede ser curva y multimodal. Se queda con el puesto
  solo si la relación entre sensores **no** es lineal.

Ambos exponen `ajustar` y `puntuar`, de modo que el script de comparación los trata igual
que al baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from torch import nn

SEMILLA = 42


@dataclass
class Mahalanobis:
    """Distancia a la nube de lo normal, teniendo en cuenta las correlaciones."""

    columnas: list[str]
    escala: StandardScaler = field(default_factory=StandardScaler)
    covarianza: LedoitWolf = field(default_factory=LedoitWolf)

    def ajustar(self, normales: pd.DataFrame) -> "Mahalanobis":
        X = self.escala.fit_transform(normales[self.columnas])
        self.covarianza.fit(X)
        return self

    def puntuar(self, tabla: pd.DataFrame) -> np.ndarray:
        X = self.escala.transform(tabla[self.columnas])
        return np.sqrt(self.covarianza.mahalanobis(X))

    def contribuciones(self, tabla: pd.DataFrame) -> pd.DataFrame:
        """Cuánto aporta cada variable a la distancia de cada ciclo.

        La distancia al cuadrado es z' S⁻¹ z, así que el término i-ésimo del producto
        z * (S⁻¹ z) reparte esa distancia entre las variables y suma exactamente el total.
        Es la explicación que el analista necesita: no «este ciclo es raro», sino «este
        ciclo es raro **por la presión PS2 en el tercer tramo**».
        """
        X = self.escala.transform(tabla[self.columnas])
        centrado = X - self.covarianza.location_
        aporte = centrado * (centrado @ self.covarianza.get_precision())
        return pd.DataFrame(aporte, index=tabla.index, columns=self.columnas)

    def principales(self, tabla: pd.DataFrame, cuantas: int = 5) -> list[list[tuple[str, float]]]:
        """Las `cuantas` variables que más explican cada ciclo, de mayor a menor."""
        aportes = self.contribuciones(tabla)
        return [sorted(fila.items(), key=lambda par: -par[1])[:cuantas]
                for _, fila in aportes.iterrows()]

    @property
    def parametros(self) -> int:
        d = len(self.columnas)
        return d + d * (d + 1) // 2


class RedVAE(nn.Module):
    """VAE pequeño a propósito: hay 172 ciclos normales para entrenar."""

    def __init__(self, entradas: int, oculta: int = 32, latente: int = 4):
        super().__init__()
        self.codificador = nn.Sequential(nn.Linear(entradas, oculta), nn.ReLU())
        self.media = nn.Linear(oculta, latente)
        self.log_varianza = nn.Linear(oculta, latente)
        self.decodificador = nn.Sequential(
            nn.Linear(latente, oculta), nn.ReLU(), nn.Linear(oculta, entradas))

    def codificar(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.codificador(x)
        return self.media(h), self.log_varianza(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.codificar(x)
        # Truco de reparametrización: muestrear sin romper el gradiente.
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var) if self.training else mu
        return self.decodificador(z), mu, log_var


def perdida_vae(reconstruido, x, mu, log_var, beta: float):
    reconstruccion = ((reconstruido - x) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)).mean()
    return reconstruccion + beta * kl, reconstruccion.item(), kl.item()


@dataclass
class DetectorVAE:
    """Entrena con lo normal y puntúa por error de reconstrucción."""

    columnas: list[str]
    latente: int = 4
    oculta: int = 32
    beta: float = 1.0
    epocas: int = 3000
    paciencia: int = 200
    lote: int = 32
    decaimiento: float = 1e-4
    escala: StandardScaler = field(default_factory=StandardScaler)
    red: RedVAE | None = None
    historial: list[dict] = field(default_factory=list)

    def ajustar(self, normales: pd.DataFrame, validacion: pd.DataFrame) -> "DetectorVAE":
        torch.manual_seed(SEMILLA)
        np.random.seed(SEMILLA)
        X = torch.tensor(self.escala.fit_transform(normales[self.columnas]),
                         dtype=torch.float32)
        V = torch.tensor(self.escala.transform(validacion[self.columnas]),
                         dtype=torch.float32)
        self.red = RedVAE(X.shape[1], self.oculta, self.latente)
        optimizador = torch.optim.Adam(self.red.parameters(), lr=1e-3,
                                       weight_decay=self.decaimiento)

        mejor, mejor_epoca, mejor_estado = float("inf"), 0, None
        for epoca in range(1, self.epocas + 1):
            self.red.train()
            orden = torch.randperm(len(X))
            for inicio in range(0, len(X), self.lote):
                lote = X[orden[inicio:inicio + self.lote]]
                optimizador.zero_grad()
                reconstruido, mu, log_var = self.red(lote)
                perdida, _, _ = perdida_vae(reconstruido, lote, mu, log_var, self.beta)
                perdida.backward()
                optimizador.step()

            self.red.eval()
            with torch.no_grad():
                salida = self.red(V)
                perdida_val, rec, kl = perdida_vae(salida[0], V, salida[1], salida[2],
                                                   self.beta)
            self.historial.append({"epoca": epoca, "perdida_validacion": float(perdida_val),
                                   "reconstruccion": rec, "kl": kl})
            if float(perdida_val) < mejor - 1e-4:
                mejor, mejor_epoca = float(perdida_val), epoca
                mejor_estado = {k: v.clone() for k, v in self.red.state_dict().items()}
            elif epoca - mejor_epoca >= self.paciencia:
                break

        if mejor_estado is not None:
            self.red.load_state_dict(mejor_estado)
        self.red.eval()
        return self

    def puntuar(self, tabla: pd.DataFrame) -> np.ndarray:
        if self.red is None:
            raise RuntimeError("El detector no está ajustado")
        X = torch.tensor(self.escala.transform(tabla[self.columnas]), dtype=torch.float32)
        with torch.no_grad():
            reconstruido, _, _ = self.red(X)
            return ((reconstruido - X) ** 2).sum(dim=1).numpy()

    @property
    def parametros(self) -> int:
        return sum(p.numel() for p in self.red.parameters()) if self.red else 0

    @property
    def mejor_epoca(self) -> int:
        if not self.historial:
            return 0
        return min(self.historial, key=lambda h: h["perdida_validacion"])["epoca"]
