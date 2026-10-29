"""Verifica los CSV de entrega contra el test, antes de armar el zip.

    python verificar_entrega.py --csv-dir entrega --test-raiz test

Reglas (las de la consigna):
  - un archivo por vertical: rd.csv, j.csv, d.csv, r.csv, it.csv
  - columnas exactamente: id, parent_uei, puesto
  - hasta 100 empresas por licitación, con puesto 1..n sin saltos ni repetidos
  - todos los id existen en licitaciones del test, y las 5 verticales cubren TODAS las licitaciones
  - toda parent_uei existe en el empresas del test de esa vertical (rankear otra empresa invalida la entrega)
  - ninguna empresa repetida dentro de una licitación
Sale con código 0 si todo está bien y 1 si hay algún error.
Para probar con un test simulado en CSV: --formato csv (lee <raiz>/<V>/licitaciones.csv y empresas.csv).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

VERTICALES = ["rd", "j", "d", "r", "it"]
COLS = ["id", "parent_uei", "puesto"]
TOP = 100


def leer(raiz: pathlib.Path, v: str, nombre: str, formato: str) -> pd.DataFrame:
    ruta = raiz / v / f"{nombre}.{'parquet' if formato == 'parquet' else 'csv'}"
    return pd.read_parquet(ruta) if formato == "parquet" else pd.read_csv(ruta)


def verificar_vertical(v: str, csv_dir: pathlib.Path, raiz: pathlib.Path, formato: str) -> list[str]:
    errores: list[str] = []
    f = csv_dir / f"{v}.csv"
    if not f.exists():
        return [f"{v}: falta el archivo {f}"]
    rk = pd.read_csv(f, dtype={"id": str, "parent_uei": str})
    if list(rk.columns) != COLS:
        errores.append(f"{v}: columnas {list(rk.columns)}; deben ser {COLS}")
        return errores
    lic = leer(raiz, v, "licitaciones", formato)
    emp = leer(raiz, v, "empresas", formato)
    ids_test = set(lic["id"].astype(str))
    emp_test = set(emp["parent_uei"].astype(str))

    if rk[COLS].isna().any().any():
        errores.append(f"{v}: hay valores vacíos")
    ajenos = set(rk["id"]) - ids_test
    if ajenos:
        errores.append(f"{v}: {len(ajenos)} id que no están en el test (ej. {sorted(ajenos)[:3]})")
    faltan = ids_test - set(rk["id"])
    if faltan:
        errores.append(f"{v}: faltan {len(faltan)} licitaciones del test sin ranking (ej. {sorted(faltan)[:3]})")
    raras = set(rk["parent_uei"]) - emp_test
    if raras:
        errores.append(f"{v}: {len(raras)} empresas que no están en empresas del test (ej. {sorted(raras)[:3]})")
    if rk.duplicated(["id", "parent_uei"]).any():
        errores.append(f"{v}: hay empresas repetidas dentro de una licitación")

    g = rk.groupby("id")["puesto"]
    mas = (g.size() > TOP).sum()
    if mas:
        errores.append(f"{v}: {mas} licitaciones con más de {TOP} filas")
    if not pd.api.types.is_integer_dtype(rk["puesto"]):
        errores.append(f"{v}: puesto no es entero")
    else:
        mal = ((g.min() != 1) | (g.max() != g.size()) | g.apply(lambda s: s.duplicated().any())).sum()
        if mal:
            errores.append(f"{v}: {mal} licitaciones con puestos que no son 1..n sin saltos ni repetidos")
    print(f"{v:>3}: {rk['id'].nunique():>5} / {len(ids_test):>5} licitaciones, {len(rk):>7,} filas, "
          f"{'OK' if not errores else 'CON ERRORES'}")
    return errores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default="entrega")
    ap.add_argument("--test-raiz", default="test")
    ap.add_argument("--formato", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--verticales", nargs="+", default=VERTICALES, help="por defecto las 5; para probar con menos, p. ej. --verticales rd")
    args = ap.parse_args()
    todos: list[str] = []
    for v in args.verticales:
        todos += verificar_vertical(v, pathlib.Path(args.csv_dir), pathlib.Path(args.test_raiz), args.formato)
    if todos:
        print("\nPROBLEMAS:")
        for e in todos:
            print(" -", e)
        sys.exit(1)
    print(f"\nTodo bien: los {len(args.verticales)} CSV cumplen las reglas de la entrega.")


if __name__ == "__main__":
    main()
