"""Descarga y verifica el conjunto Condition Monitoring of Hydraulic Systems (UCI 447).

No versionamos los datos: este script los obtiene y calcula la huella SHA-256 del archivo
descargado y de cada archivo de sensor, de modo que cualquiera pueda comprobar que trabajó
sobre exactamente los mismos bytes.

Fuente: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
Licencia: Creative Commons Attribution 4.0 International (CC BY 4.0)
DOI: 10.24432/C5CW21

Uso:
    uv run python scripts/download_data.py
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = ("https://archive.ics.uci.edu/static/public/447/"
       "condition+monitoring+of+hydraulic+systems.zip")
RAIZ = Path(__file__).resolve().parents[1]
CRUDO = RAIZ / "data" / "crudo"
HUELLAS = RAIZ / "data" / "huellas.json"

# Los 17 sensores del banco, con su frecuencia de muestreo en Hz y su unidad.
SENSORES = {
    "PS1": (100, "bar"), "PS2": (100, "bar"), "PS3": (100, "bar"),
    "PS4": (100, "bar"), "PS5": (100, "bar"), "PS6": (100, "bar"),
    "EPS1": (100, "W"),
    "FS1": (10, "l/min"), "FS2": (10, "l/min"),
    "TS1": (1, "degC"), "TS2": (1, "degC"), "TS3": (1, "degC"), "TS4": (1, "degC"),
    "VS1": (1, "mm/s"),
    "CE": (1, "%"), "CP": (1, "kW"), "SE": (1, "%"),
}
PERFIL = "profile"


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            digest.update(bloque)
    return digest.hexdigest()


def descargar(destino: Path) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        print(f"{destino.name} ya estaba descargado")
        return destino
    print(f"Descargando {URL}")
    peticion = Request(URL, headers={"User-Agent": "INF8239-TF/1.0 (uso academico)"})
    with urlopen(peticion, timeout=180) as respuesta, destino.open("wb") as archivo:
        archivo.write(respuesta.read())
    print(f"{destino.name}: {destino.stat().st_size / 1e6:.1f} MB")
    return destino


def main() -> None:
    comprimido = descargar(CRUDO / "hydraulic.zip")

    with zipfile.ZipFile(comprimido) as zf:
        nombres = zf.namelist()
        zf.extractall(CRUDO)
    print(f"Extraídos {len(nombres)} archivos en {CRUDO}")

    esperados = [f"{s}.txt" for s in SENSORES] + [f"{PERFIL}.txt"]
    faltan = [n for n in esperados if not (CRUDO / n).exists()]
    if faltan:
        raise SystemExit(f"Faltan archivos esperados: {faltan}\nEl zip trae: {nombres}")

    huellas = {"url": URL, "licencia": "CC BY 4.0", "doi": "10.24432/C5CW21",
               "zip_sha256": sha256(comprimido), "archivos": {}}
    for nombre in esperados:
        ruta = CRUDO / nombre
        huellas["archivos"][nombre] = {
            "sha256": sha256(ruta),
            "bytes": ruta.stat().st_size,
        }
    HUELLAS.write_text(json.dumps(huellas, indent=2), encoding="utf-8")

    print(f"\nHuellas guardadas en {HUELLAS.relative_to(RAIZ)}")
    for nombre, datos in list(huellas["archivos"].items())[:3]:
        print(f"  {nombre:<12} {datos['sha256'][:16]}  {datos['bytes'] / 1e6:6.2f} MB")
    print(f"  ... y {len(esperados) - 3} archivos más")


if __name__ == "__main__":
    main()
