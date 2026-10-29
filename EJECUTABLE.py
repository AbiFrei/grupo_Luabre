"""Corre toda la entrega de punta a punta, sin Colab: datos, test, vectores, modelo y verificación.

    python EJECUTABLE.py                 # desde la raíz del repo; escribe entrega/rd.csv, d.csv, it.csv, r.csv y j.csv
    python EJECUTABLE.py --salida-dir otra_carpeta

Hace lo mismo que los comandos del README (y que el notebook de Colab):
  1. instala las librerías de requirements.txt (salvo con --sin-pip),
  2. baja los datos y el test de la cátedra y los vectores MiniLM del historial (con reintentos y comprobando que no queden cortados),
  3. corre `modelo_ranker_texto.py --todas --modo final` (entrena y escribe los cinco CSV),
  4. verifica los CSV con `verificar_entrega.py` y muestra su huella md5.
Necesita Python 3.10+, conexión a internet y, de preferencia, una GPU (con GPU tarda unos 25 minutos; sin GPU, bastante más).
Si no se pueden bajar los vectores, el modelo los calcula solo (más de 1 hora en una GPU).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

DATOS_URL = "https://github.com/bravoivanagustin/ranking-licitaciones/releases/download/datos-v1/data.zip"
TEST_URL = "https://github.com/bravoivanagustin/ranking-licitaciones/releases/download/test-v1/test.zip"
VECTORES_URL = "https://github.com/AbiFrei/grupo_Luabre/releases/download/vectores-v1"
FILAS_HIST = {"rd": 84835, "d": 256709, "it": 472455, "r": 811685, "j": 273416}   # contratos del historial por vertical
VERTICALES = ["rd", "j", "d", "r", "it"]


def bajar(url: str, destino: pathlib.Path, bytes_esperados: int | None = None, intentos: int = 5) -> bool:
    """Baja url a destino; si queda cortado (tamaño distinto del esperado o del Content-Length) reintenta desde cero."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, intentos + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(destino, "wb") as f:
                esperado = bytes_esperados or int(r.headers.get("Content-Length") or 0) or None
                shutil.copyfileobj(r, f, 1 << 20)
            if esperado is None or destino.stat().st_size == esperado:
                return True
            print(f"  {destino.name}: quedó cortado ({destino.stat().st_size} de {esperado} bytes), reintento {i}/{intentos}")
        except Exception as e:  # red, 404, etc.
            print(f"  {destino.name}: {type(e).__name__}: {e} (intento {i}/{intentos})")
        time.sleep(3)
    destino.unlink(missing_ok=True)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--salida-dir", default="entrega")
    ap.add_argument("--sin-pip", action="store_true", help="no instalar requirements.txt (ya está el entorno)")
    ap.add_argument("--sin-vectores", action="store_true", help="no bajar los vectores: el modelo los calcula (más de 1 hora)")
    ap.add_argument("--solo-descargas", action="store_true", help="solo baja los archivos y termina (para probar)")
    a = ap.parse_args()
    raiz = pathlib.Path(__file__).resolve().parent
    os.chdir(raiz)

    if not a.sin_pip:
        print("== 1. Librerías (requirements.txt)")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

    print("== 2. Datos, test y vectores")
    for nombre, url, carpeta in (("data", DATOS_URL, "data"), ("test", TEST_URL, "test")):
        if pathlib.Path(carpeta).is_dir():
            print(f"  {carpeta}/ ya está")
            continue
        z = pathlib.Path(f"{nombre}.zip")
        if not bajar(url, z):
            sys.exit(f"No se pudo bajar {url}. Revisá la conexión y volvé a correr.")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(".")
        z.unlink()
        print(f"  {carpeta}/ listo")
    if not a.sin_vectores:
        for v, filas in FILAS_HIST.items():
            f = pathlib.Path("cache") / f"emb_hist_{v}.npy"
            esperado = filas * 384 * 4 + 128
            if f.exists() and f.stat().st_size == esperado:
                print(f"  {f.name}: OK")
            elif bajar(f"{VECTORES_URL}/{f.name}", f, esperado):
                print(f"  {f.name}: OK")
            else:
                print(f"  ATENCIÓN {f.name}: no se pudo bajar; el modelo lo calcula al correr (más de 1 hora)")
    if a.solo_descargas:
        return

    print("== 3. Modelo (entrena y escribe los cinco CSV)")
    subprocess.run([sys.executable, "modelo_ranker_texto.py", "--todas", "--cache", "cache", "--modo", "final",
                    "--test-raiz", "test", "--salida-dir", a.salida_dir], check=True)

    print("== 4. Verificación")
    subprocess.run([sys.executable, "verificar_entrega.py", "--csv-dir", a.salida_dir, "--test-raiz", "test"], check=True)
    print("\nHuellas md5:")
    for v in VERTICALES:
        p = pathlib.Path(a.salida_dir) / f"{v}.csv"
        print(f"  {p}  {hashlib.md5(p.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
