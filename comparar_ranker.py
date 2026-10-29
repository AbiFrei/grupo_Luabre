"""Compara el ranker del historial (LightGBM, rankings que nos pasó el grupo) con B0 y con nuestra mezcla, en 2024S1.

Los tres se miden con el mismo `evaluar.por_licitacion` (MRR filtrado) sobre las mismas licitaciones.
Para nuestra mezcla y para B0 usamos los archivos por licitación que escribió experimento_emb.py
(`por_licitacion_emb<etiqueta>_<V>.csv`, etapa "prueba"); para el ranker, su `ranking_<V>.csv`.
Mostramos el MRR por vertical como referencia y sacamos las conclusiones con las cinco verticales juntas
(IC95 por bootstrap pareado sobre las licitaciones).

Uso:

    python comparar_ranker.py --rankings rankings_grupo --resultados resultados
    python comparar_ranker.py --rankings rankings_grupo --resultados resultados --etiqueta _ext --alpha 1.5
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

import evaluar as ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rankings", default="rankings_grupo", help="carpeta con ranking_<V>.csv")
    ap.add_argument("--resultados", default="resultados", help="carpeta con por_licitacion_emb<etiqueta>_<V>.csv")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--etiqueta", default="_ext")
    ap.add_argument("--alpha", default="1.5", help="alpha global de nuestra mezcla")
    ap.add_argument("--verticales", default="rd,d,it,r,j")
    ap.add_argument("--rankings-texto", default=None, help="carpeta con ranking_<V>.csv del ranker del historial + texto (opcional)")
    a = ap.parse_args()

    partes = []
    filas = []
    for V in a.verticales.split(","):
        rk = pd.read_csv(pathlib.Path(a.rankings) / f"ranking_{V}.csv")
        gan = pd.read_parquet(pathlib.Path(a.datos) / V / "ganadores.parquet")
        ids = set(rk.id)
        g = gan[gan.id.isin(ids)]
        ranker = ev.por_licitacion(rk, g)

        x = pd.read_csv(pathlib.Path(a.resultados) / f"por_licitacion_emb{a.etiqueta}_{V}.csv", dtype={"alpha": str})
        p = x[x.etapa == "prueba"]
        b0 = p[p.variante == "B0"].set_index("id")
        nuestra = p[(p.variante == "mezcla") & (p.alpha == a.alpha)].set_index("id")

        rtexto = None
        if a.rankings_texto:
            rkt = pd.read_csv(pathlib.Path(a.rankings_texto) / f"ranking_{V}.csv")
            rtexto = ev.por_licitacion(rkt[rkt.id.isin(ids)], g)

        comun = ranker.index.intersection(b0.index).intersection(nuestra.index)
        if rtexto is not None:
            comun = comun.intersection(rtexto.index)
        t = pd.DataFrame({"B0": b0.loc[comun, "MRR"], "nuestra": nuestra.loc[comun, "MRR"],
                          "ranker": ranker.loc[comun, "MRR"], "recall_nuestra": nuestra.loc[comun, "recall"],
                          "recall_ranker": ranker.loc[comun, "recall"], "hit5_nuestra": nuestra.loc[comun, "Hit@5"],
                          "hit5_ranker": ranker.loc[comun, "Hit@5"]})
        fila = {"vertical": V, "N": len(comun), "licitaciones del ranker": len(ranker),
                "B0": t.B0.mean(), "nuestra mezcla": t.nuestra.mean(), "ranker del historial": t.ranker.mean()}
        if rtexto is not None:
            t["ranker_texto"] = rtexto.loc[comun, "MRR"]
            t["recall_texto"] = rtexto.loc[comun, "recall"]
            t["hit5_texto"] = rtexto.loc[comun, "Hit@5"]
            fila["ranker + texto"] = t.ranker_texto.mean()
        t["vertical"] = V
        partes.append(t)
        filas.append(fila)

    todo = pd.concat(partes)
    cols = ["B0", "nuestra", "ranker"] + (["ranker_texto"] if a.rankings_texto else [])
    juntas = {"vertical": "juntas", "N": len(todo), "licitaciones del ranker": "",
              "B0": todo.B0.mean(), "nuestra mezcla": todo.nuestra.mean(), "ranker del historial": todo.ranker.mean()}
    macro = todo.groupby("vertical")[cols].mean().mean()
    prom = {"vertical": "promedio de verticales", "N": "", "licitaciones del ranker": "",
            "B0": macro.B0, "nuestra mezcla": macro.nuestra, "ranker del historial": macro.ranker}
    if a.rankings_texto:
        juntas["ranker + texto"] = todo.ranker_texto.mean()
        prom["ranker + texto"] = macro.ranker_texto
    filas += [juntas, prom]

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print("MRR en prueba (2024S1), mismas licitaciones:")
        print(pd.DataFrame(filas).to_string(index=False))
        print("\nRecall@100 y Hit@5 (todas las licitaciones juntas):")
        print(f"nuestra mezcla:       Recall@100 {todo.recall_nuestra.mean():.3f} | Hit@5 {todo.hit5_nuestra.mean():.3f}")
        print(f"ranker del historial: Recall@100 {todo.recall_ranker.mean():.3f} | Hit@5 {todo.hit5_ranker.mean():.3f}")
        if a.rankings_texto:
            print(f"ranker + texto:       Recall@100 {todo.recall_texto.mean():.3f} | Hit@5 {todo.hit5_texto.mean():.3f}")

    print("\nΔ MRR, todas las licitaciones juntas (IC95 por bootstrap pareado):")
    pares = [("ranker del historial", "B0", todo.ranker, todo.B0), ("nuestra mezcla", "B0", todo.nuestra, todo.B0),
             ("ranker del historial", "nuestra mezcla", todo.ranker, todo.nuestra)]
    if a.rankings_texto:
        pares += [("ranker + texto", "B0", todo.ranker_texto, todo.B0),
                  ("ranker + texto", "nuestra mezcla", todo.ranker_texto, todo.nuestra),
                  ("ranker + texto", "ranker del historial", todo.ranker_texto, todo.ranker)]
    for nom_x, nom_y, x_, y_ in pares:
        lo, hi = ev.bootstrap_delta(x_.values, y_.values)
        print(f"{nom_x:21s} contra {nom_y:21s}: Δ {np.mean(x_.values - y_.values):+.4f}, IC95 ({lo:+.4f}, {hi:+.4f})")


if __name__ == "__main__":
    main()
