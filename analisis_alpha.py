"""Alpha global contra alpha por vertical contra dejar una vertical afuera.

Usa los archivos por licitación que escribe experimento_emb.py (`por_licitacion_emb<etiqueta>_<V>.csv`), así que
no recalcula nada: solo compara tres formas de elegir el alpha de la mezcla (TF-IDF + embeddings):

    global      un solo alpha, elegido en selección (2023) con las licitaciones de todas las verticales juntas
    por vertical  un alpha por vertical, elegido en selección con las licitaciones de esa vertical
    sin la vertical  para cada vertical V, el alpha elegido en selección con las OTRAS verticales (V no se mira)
                 → mide qué tan bien funciona el método en una vertical que no vio al elegir el alpha

En todos los casos el alpha es el de mejor MRR en selección y se mide en prueba (2024S1). Como referencia mostramos también el
mejor alpha mirando la prueba (el techo), que NO es una regla utilizable: es el techo de lo que podría ganar cualquier regla.

Uso:

    python analisis_alpha.py
    python analisis_alpha.py --dir resultados --etiqueta _ext --variante mezcla
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

import evaluar as ev


def cargar(dir_, etiqueta, variante, verticales):
    """{etapa: DataFrame(filas = (vertical, id), columnas = alpha, valores = MRR)} y lo mismo para B0."""
    tablas, b0 = {}, {}
    for V in verticales:
        x = pd.read_csv(pathlib.Path(dir_) / f"por_licitacion_emb{etiqueta}_{V}.csv", dtype={"alpha": str})
        for etapa in ("seleccion", "prueba"):
            e = x[x.etapa == etapa]
            m = e[e.variante == variante].assign(alpha=lambda d: d.alpha.astype(float))
            t = m.pivot_table(index="id", columns="alpha", values="MRR", aggfunc="first")
            t.index = pd.MultiIndex.from_product([[V], t.index])
            tablas.setdefault(etapa, []).append(t)
            bb = e[e.variante == "B0"].set_index("id")["MRR"]
            bb.index = pd.MultiIndex.from_product([[V], bb.index])
            b0.setdefault(etapa, []).append(bb)
    return ({e: pd.concat(v) for e, v in tablas.items()}, {e: pd.concat(v) for e, v in b0.items()})


def mejor_alpha(tabla: pd.DataFrame, macro: bool = False) -> float:
    """Alpha con mejor MRR medio; `macro` promedia primero dentro de cada vertical y después entre verticales."""
    if macro:
        puntaje = tabla.groupby(level=0).mean().mean()
    else:
        puntaje = tabla.mean()
    return float(puntaje.idxmax())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="resultados")
    ap.add_argument("--etiqueta", default="_ext", help="la misma que usamos en experimento_emb.py")
    ap.add_argument("--variante", default="mezcla", help="tfidf, emb o mezcla")
    ap.add_argument("--verticales", default="rd,d,it,r,j")
    a = ap.parse_args()
    V_ = a.verticales.split(",")
    tab, b0 = cargar(a.dir, a.etiqueta, a.variante, V_)
    sel, pru = tab["seleccion"], tab["prueba"]
    n_sel = sel.groupby(level=0).size()
    n_pru = pru.groupby(level=0).size()
    print(f"Variante: {a.variante} | alphas: {list(sel.columns)}")
    print("Licitaciones en selección:", dict(n_sel), "| en prueba:", dict(n_pru))

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        t = sel.groupby(level=0).mean().T
        t["juntas"] = sel.mean()
        t["promedio verticales"] = sel.groupby(level=0).mean().mean()
        print("\nMRR en selección (2023) por alpha:")
        print(t.reindex(columns=V_ + ["juntas", "promedio verticales"]).rename_axis("alpha").to_string())

    # --- alphas elegidos en selección ---
    a_glob = mejor_alpha(sel)
    a_glob_macro = mejor_alpha(sel, macro=True)
    a_vert = {V: mejor_alpha(sel.loc[V]) for V in V_}
    a_sin = {V: mejor_alpha(sel.drop(index=V, level=0)) for V in V_}
    a_sin_macro = {V: mejor_alpha(sel.drop(index=V, level=0), macro=True) for V in V_}
    a_orac = {V: mejor_alpha(pru.loc[V]) for V in V_}
    print(f"\nalpha global (licitaciones juntas): {a_glob} | alpha global (promedio de verticales): {a_glob_macro}")
    print("alpha por vertical:       ", a_vert)
    print("alpha sin la vertical:    ", a_sin, "| promediando verticales:", a_sin_macro)
    print("alpha techo (mirando la prueba, no utilizable):", a_orac)

    # --- prueba ---
    def vec(regla: dict | float) -> pd.Series:
        partes = []
        for V in V_:
            al = regla[V] if isinstance(regla, dict) else regla
            partes.append(pru.loc[V][al].set_axis(pd.MultiIndex.from_product([[V], pru.loc[V].index])))
        return pd.concat(partes)

    reglas = {"B0": b0["prueba"].reindex(pru.index),
              "alpha global": vec(a_glob),
              "alpha por vertical": vec(a_vert),
              "alpha sin la vertical": vec(a_sin),
              "techo (no utilizable)": vec(a_orac)}
    filas = {nombre: [v.loc[V].mean() for V in V_] + [v.mean(), np.mean([v.loc[V].mean() for V in V_])]
             for nombre, v in reglas.items()}
    cols = V_ + ["juntas", "promedio verticales"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print("\nMRR en prueba (2024S1) según cómo elegimos el alpha:")
        print(pd.DataFrame(filas, index=cols).T.to_string())

    print("\nΔ MRR en prueba (todas las licitaciones juntas), IC95 por bootstrap pareado:")
    pares = [("alpha global", "B0"), ("alpha por vertical", "B0"), ("alpha sin la vertical", "B0"),
             ("alpha por vertical", "alpha global"), ("alpha sin la vertical", "alpha global"),
             ("alpha sin la vertical", "alpha por vertical")]
    for x, y in pares:
        d = reglas[x].values - reglas[y].values
        lo, hi = ev.bootstrap_delta(reglas[x].values, reglas[y].values)
        print(f"{x:22s} contra {y:20s}: Δ MRR {d.mean():+.4f}, IC95 ({lo:+.4f}, {hi:+.4f})")

    print("\nΔ MRR en prueba por vertical: alpha por vertical contra alpha global / sin la vertical contra alpha global")
    for V in V_:
        out = []
        for x in ("alpha por vertical", "alpha sin la vertical"):
            a_, b_ = reglas[x].loc[V].values, reglas["alpha global"].loc[V].values
            lo, hi = ev.bootstrap_delta(a_, b_)
            out.append(f"{x}: {np.mean(a_ - b_):+.4f} ({lo:+.4f}, {hi:+.4f})")
        print(f"{V:3s} (alpha por vertical {a_vert[V]}, sin la vertical {a_sin[V]}, global {a_glob}) | " + " | ".join(out))


if __name__ == "__main__":
    main()
