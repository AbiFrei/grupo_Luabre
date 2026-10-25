"""Experimento del "top protegido": respetamos el top-k de B0 y usamos el texto solo para ordenar el resto.

La mezcla (B0 + TF-IDF + embeddings) mejora el MRR, pero en las licitaciones donde B0 ya tenía al
ganador entre sus 10 primeras (S_inc) baja el Hit@5 (rd, it). Acá probamos la variante que protege el
principio del ranking de B0:

    k = 0    la mezcla de siempre: norm(p) + alpha * (norm(sim_tfidf) + 0.5 * norm(sim_emb))
    k > 0    los k primeros puestos son los de B0 (ordenados solo por p) y del puesto k+1 en adelante
             ordenamos al resto de las candidatas con el puntaje de la mezcla

Para cada k elegimos el alpha con mejor MRR en selección (2023) y lo medimos en prueba (2024S1) contra
B0 y contra la mezcla sin proteger (k = 0), con IC95 por bootstrap pareado.

Uso:

    python experimento_top.py rd
    python experimento_top.py rd d it r j --ks 0,3,5,10 --alphas 0.5,1,1.5,2,3

Usa los vectores guardados en `--cache` por experimento_emb.py (con MiniLM); si faltan, los calcula
(necesita GPU). Escribe, en `--salida`: resumen_top_<V>.csv y por_licitacion_top_<V>.csv.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

import b0
import evaluar as ev
import experimento as base
import experimento_emb as emb

KS = [0, 3, 5, 10]
ALPHAS = [0.5, 1, 1.5, 2, 3]


def ranking_top(sc: dict, alpha: float, k: int, top: int = 100, seed: int = 0) -> pd.DataFrame:
    """Ranking con los k primeros puestos de B0 y el resto ordenado por la mezcla.

    Con k = 0 es idéntico a `experimento.ranking` (mismo ruido de desempate, mismo orden de licitaciones).
    """
    rng = np.random.default_rng(seed)
    filas = []
    for lid, (ids, p, sim) in sc.items():
        ruido = 1e-9 * rng.random(len(p))
        orden = np.argsort(-(base.norm(p) + alpha * base.norm(sim) + ruido))
        if k > 0:
            fijos = np.argsort(-(base.norm(p) + ruido))[:k]
            orden = np.concatenate([fijos, orden[~np.isin(orden, fijos)]])
        for puesto, i in enumerate(orden[:top], 1):
            filas.append((lid, ids[i], puesto))
    return pd.DataFrame(filas, columns=["id", "parent_uei", "puesto"])


def medir_top(sc, alpha, k, lics, lic, emp, gan, rk_b0):
    """Devuelve (resumen por conjunto, métricas por licitación) del ranking con ese k y ese alpha."""
    g = gan[gan.id.isin(lics.id)]
    rk = ranking_top(sc, alpha, k)
    rk = rk[rk.id.isin(set(g.id))]
    rk0 = rk_b0[rk_b0.id.isin(set(g.id))]
    return ev.evaluar(rk, lic, emp, g, rk0), ev.por_licitacion(rk, g)


def comparar_top(sc, a_n, k_n, a_r, k_r, g) -> dict:
    """Δ MRR del ranking (k_n, a_n) contra el (k_r, a_r), con IC95 por bootstrap pareado."""
    def mrr(a, k):
        rk = ranking_top(sc, a, k)
        return ev.por_licitacion(rk[rk.id.isin(set(g.id))], g)["MRR"]
    x, y = mrr(a_n, k_n).align(mrr(a_r, k_r), join="inner")
    lo, hi = ev.bootstrap_delta(x.values, y.values)
    return {"Δ MRR": x.mean() - y.mean(), "IC95 bajo": lo, "IC95 alto": hi,
            "mejoran": int((x > y).sum()), "empeoran": int((x < y).sum()), "iguales": int((x == y).sum())}


def vectores(V, hist, lic, modelo, cache) -> tuple:
    """Vectores de los contratos (con caché, igual que experimento_emb.py) y de las licitaciones."""
    cache = pathlib.Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    ruta = cache / f"emb_hist_{V}.npy"
    emb_hist = np.load(ruta) if ruta.exists() else None
    if emb_hist is None or len(emb_hist) != len(hist):
        t0 = time.time()
        emb_hist = emb.codificar(modelo, hist["descripcion"].fillna(""))
        np.save(ruta, emb_hist)
        print(f"Vectores de {len(hist)} contratos: {time.time() - t0:.0f} s", flush=True)
    else:
        print(f"Vectores de contratos leídos de {ruta}", flush=True)
    emb_lic = emb.codificar(modelo, lic["title"].map(base.txt) + " " + lic["description"].map(base.txt))
    return emb_hist, emb_lic, dict(zip(lic["id"], range(len(lic))))


def correr_vertical(V, a, modelo) -> None:
    t_ini = time.time()
    d = pathlib.Path(a.datos) / V
    print(f"\n=== Vertical {V} (top protegido) ===", flush=True)
    lic = pd.read_parquet(d / "licitaciones.parquet")
    gan = pd.read_parquet(d / "ganadores.parquet")
    emp = pd.read_parquet(d / "empresas.parquet")
    hist = pd.read_parquet(d / "historial.parquet")

    ruta_b0 = pathlib.Path(a.b0dir) / f"b0_{V}.csv"
    if not ruta_b0.exists():
        subprocess.run([sys.executable, "b0.py", str(d), "--salida", str(ruta_b0)], check=True)
    rk_b0 = pd.read_csv(ruta_b0)

    sel = lic[(lic.t >= base.SEL_DESDE) & (lic.t < base.PRUEBA_DESDE)].copy()
    pru = lic[lic.t >= base.PRUEBA_DESDE].copy()
    print(f"selección: {len(sel)} licitaciones | prueba: {len(pru)} licitaciones", flush=True)

    emb_hist, emb_lic, fila_lic = vectores(V, hist, lic, modelo, a.cache)
    prior = b0.B0(hist.copy())
    mezcla = {}
    for etapa, lics in (("seleccion", sel), ("prueba", pru)):
        print(f"Puntajes de {etapa}:", flush=True)
        sc_t = base.puntuar(lics, emp, hist, prior)
        sc_e = emb.puntuar_emb(lics, sc_t, emp, hist, emb_hist, emb_lic, fila_lic)
        mezcla[etapa] = emb.mezclar(sc_t, sc_e, emb.W_MEZCLA)

    filas, por_lic, mrr_sel = [], [], {}
    for etapa, lics in (("seleccion", sel), ("prueba", pru)):
        g = gan[gan.id.isin(lics.id)]
        for k in a.ks:
            for al in a.alphas:
                res, pl = medir_top(mezcla[etapa], al, k, lics, lic, emp, gan, rk_b0)
                filas.append(res.assign(vertical=V, etapa=etapa, k=k, alpha=al))
                por_lic.append(pl.assign(vertical=V, etapa=etapa, k=k, alpha=al).reset_index())
                if etapa == "seleccion":
                    mrr_sel[(k, al)] = float(res.loc[res.conjunto == "todas", "MRR"].iloc[0])
        print(f"Etapa {etapa} medida ({time.time() - t_ini:.0f} s desde el inicio)", flush=True)

    resumen = pd.concat(filas, ignore_index=True)
    sal = pathlib.Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(sal / f"resumen_top_{V}.csv", index=False)
    pd.concat(por_lic, ignore_index=True).to_csv(sal / f"por_licitacion_top_{V}.csv", index=False)

    # alpha elegido en selección para cada k
    elegido = {k: max(a.alphas, key=lambda al: mrr_sel[(k, al)]) for k in a.ks}
    k_ref = 0 if 0 in a.ks else a.ks[0]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print("\nMRR en selección (2023), todas las licitaciones (filas: alpha, columnas: k):")
        print(pd.DataFrame({f"k={k}": [mrr_sel[(k, al)] for al in a.alphas] for k in a.ks},
                           index=pd.Index(a.alphas, name="alpha")).to_string())
        print("\nalpha elegido en selección por k: " + " | ".join(f"k={k}: {elegido[k]}" for k in a.ks))

        r = resumen[resumen.etapa == "prueba"]
        r = pd.concat([r[(r.k == k) & (r.alpha == elegido[k])] for k in a.ks])
        print("\nPrueba (2024S1) contra B0, por conjunto:")
        print(r[["k", "alpha", "conjunto", "N", "MRR", "B0 MRR", "Δ MRR", "Δ MRR IC95", "Hit@5", "B0 Hit@5"]]
              .to_string(index=False))

        print(f"\nContra la mezcla sin proteger (k={k_ref}, alpha {elegido[k_ref]}), todas las licitaciones:")
        for k in a.ks:
            if k == k_ref:
                continue
            for etapa, lics in (("seleccion", sel), ("prueba", pru)):
                c = comparar_top(mezcla[etapa], elegido[k], k, elegido[k_ref], k_ref,
                                 gan[gan.id.isin(lics.id)])
                print(f"k={k:2d} (alpha {elegido[k]}) {etapa:9s}: Δ MRR {c['Δ MRR']:+.3f}, "
                      f"IC95 ({c['IC95 bajo']:+.3f}, {c['IC95 alto']:+.3f}) | mejoran {c['mejoran']}, "
                      f"empeoran {c['empeoran']}, iguales {c['iguales']}")
    print(f"\nTiempo total {V}: {time.time() - t_ini:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("verticales", nargs="+", help="rd, d, it, r y/o j")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--b0dir", default=".")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--ks", default=",".join(str(x) for x in KS), help="cuántos puestos protegemos de B0")
    ap.add_argument("--alphas", default=",".join(str(x) for x in ALPHAS))
    a = ap.parse_args()
    a.ks = [int(x) for x in a.ks.split(",")]
    a.alphas = [float(x) for x in a.alphas.split(",")]
    print(f"k: {a.ks} | alphas: {a.alphas}", flush=True)
    modelo = emb.cargar_modelo()
    for V in a.verticales:
        try:
            correr_vertical(V, a, modelo)
        except Exception:
            print(f"\n*** Falló la vertical {V}:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    main()
