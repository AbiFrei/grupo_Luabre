"""Compara dos carpetas de rankings de validación (modo A y modo B) contra los ganadores reales.

    python comparar_modos.py --a val23_A --b val23_B --corte 2023-07-01 --datos data

Calcula, por vertical y con todas juntas, el MRR de B0, A y B sobre las licitaciones con t >= corte que tienen ganador,
y la diferencia B menos A con IC95 por bootstrap pareado (remuestreando licitaciones). Si --corte es anterior a 2024-01-01,
repite el cálculo solo con las licitaciones anteriores a 2024-01-01 (ventana distinta de 2024S1).
Lee de cada carpeta ranking_<V>.csv y b0_val_<V>.csv, que deja modelo_ranker_texto.py en modo validar.
"""
from __future__ import annotations
import argparse, pathlib
import numpy as np, pandas as pd

VERT = ["rd", "d", "it", "r", "j"]


def rr_por_lic(rank, gd, ids):
    g = rank[[u in gd.get(i, ()) for i, u in zip(rank["id"], rank["parent_uei"])]]
    mejor = g.groupby("id")["puesto"].min()
    s = pd.Series(0.0, index=ids)
    com = mejor.index.intersection(s.index)
    s.loc[com] = 1.0 / mejor.loc[com]
    return s


def ic(d, rng, B=5000):
    d = np.asarray(d, float)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--corte", default="2023-07-01"); ap.add_argument("--datos", default="data")
    ap.add_argument("--formato", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--verticales", nargs="+", default=VERT)
    a = ap.parse_args()
    leer = pd.read_parquet if a.formato == "parquet" else pd.read_csv
    ext = "parquet" if a.formato == "parquet" else "csv"
    rng = np.random.default_rng(0)
    filas = []
    for V in a.verticales:
        lic = leer(pathlib.Path(a.datos) / V / f"licitaciones.{ext}")[["id", "t"]]
        gan = leer(pathlib.Path(a.datos) / V / f"ganadores.{ext}")
        lic["id"] = lic["id"].astype(str); gan["id"] = gan["id"].astype(str)
        lic["t"] = pd.to_datetime(lic["t"])
        gd = gan.groupby("id")["parent_uei"].apply(set).to_dict()
        val = lic[(lic["t"] >= a.corte) & lic["id"].isin(gd)]
        ids = val["id"].tolist()
        cols = {}
        for nom, carpeta, arch in [("A", a.a, f"ranking_{V}.csv"), ("B", a.b, f"ranking_{V}.csv"), ("B0", a.a, f"b0_val_{V}.csv")]:
            r = pd.read_csv(pathlib.Path(carpeta) / arch); r["id"] = r["id"].astype(str)
            cols[nom] = rr_por_lic(r, gd, ids)
        df = pd.DataFrame(cols)
        df["t"] = val.set_index("id")["t"].reindex(df.index)
        df["V"] = V
        filas.append(df)
    todo = pd.concat(filas)
    def informe(d, titulo):
        print(f"\n=== {titulo} ===")
        print(f"{'vertical':8s} {'N':>5s} {'B0':>7s} {'A':>7s} {'B':>7s}   B menos A (IC95)")
        for V in a.verticales + ["JUNTAS"]:
            x = d if V == "JUNTAS" else d[d.V == V]
            if len(x) == 0: continue
            m, lo, hi = ic(x.B - x.A, rng)
            print(f"{V:8s} {len(x):5d} {x.B0.mean():7.4f} {x.A.mean():7.4f} {x.B.mean():7.4f}   {m:+.4f} ({lo:+.4f} a {hi:+.4f})")
        m, lo, hi = ic(d.B - d.A, rng)
        print("Regla: B se adopta si el IC95 de las 5 verticales juntas es todo positivo ->", "SE CUMPLE" if lo > 0 else "NO se cumple")
    informe(todo, f"todas las licitaciones con t >= {a.corte}")
    if pd.Timestamp(a.corte) < pd.Timestamp("2024-01-01"):
        informe(todo[todo.t < "2024-01-01"], f"solo {a.corte} a 2023-12-31 (ventana distinta de 2024S1)")
        informe(todo[todo.t >= "2024-01-01"], "solo 2024S1")

if __name__ == "__main__":
    main()
