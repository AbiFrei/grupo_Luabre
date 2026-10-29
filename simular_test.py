"""Arma un TEST SIMULADO con el mismo formato que el test oficial, a partir de data/ (solo para ensayar el flujo).

    python simular_test.py --salida test_sim            # las 5 verticales
    python simular_test.py --salida test_sim --datos data/rd

Para cada vertical deja <salida>/<V>/licitaciones.parquet y empresas.parquet:
  - licitaciones: las de t >= --desde (por defecto 2024-01-01), con id nuevo <V>-00001, ... y SIN ganadores
  - empresas: una sola foto (la del semestre de --desde) SIN la columna `semestre`
Guarda además <salida>/<V>/_ganadores_ocultos.csv (id nuevo, parent_uei) por si se quiere medir; el modelo no lo lee.
OJO: como las licitaciones simuladas también están en data/, el modelo las ve al entrenar. Sirve para comprobar que el
flujo corre y que el formato es válido, NO para medir el MRR (para eso está el modo `validar`).
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd


def semestre_de(fecha: pd.Timestamp) -> str:
    return f"{fecha.year}S{1 if fecha.month <= 6 else 2}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", nargs="+")
    ap.add_argument("--salida", default="test_sim")
    ap.add_argument("--desde", default="2024-01-01")
    ap.add_argument("--formato", choices=["parquet", "csv"], default="parquet")
    a = ap.parse_args()
    dirs = a.datos or sorted(str(p.parent) for p in pathlib.Path("data").glob("*/licitaciones.parquet"))
    desde = pd.Timestamp(a.desde)
    sem = semestre_de(desde)
    leer = pd.read_parquet if a.formato == "parquet" else pd.read_csv
    for d in dirs:
        d = pathlib.Path(d)
        v = d.name
        ext = "parquet" if a.formato == "parquet" else "csv"
        lic = leer(d / f"licitaciones.{ext}")
        emp = leer(d / f"empresas.{ext}")
        gan = leer(d / f"ganadores.{ext}")
        lic["t"] = pd.to_datetime(lic["t"])
        lic = lic[lic["t"] >= desde].sort_values(["t", "id"], kind="stable").reset_index(drop=True)
        nuevo = [f"{v}-{i + 1:05d}" for i in range(len(lic))]
        mapa = dict(zip(lic["id"], nuevo))
        lic["id"] = nuevo
        snap = emp[emp["semestre"] == sem].drop(columns="semestre").drop_duplicates("parent_uei").reset_index(drop=True)
        g = gan[gan["id"].isin(mapa)].assign(id=lambda x: x["id"].map(mapa))[["id", "parent_uei"]]
        out = pathlib.Path(a.salida) / v
        out.mkdir(parents=True, exist_ok=True)
        if a.formato == "parquet":
            lic.to_parquet(out / "licitaciones.parquet", index=False)
            snap.to_parquet(out / "empresas.parquet", index=False)
        else:
            lic.to_csv(out / "licitaciones.csv", index=False)
            snap.to_csv(out / "empresas.csv", index=False)
        g.to_csv(out / "_ganadores_ocultos.csv", index=False)
        print(f"{v}: {len(lic):,} licitaciones, {len(snap):,} empresas (semestre {sem}) -> {out}")


if __name__ == "__main__":
    main()
