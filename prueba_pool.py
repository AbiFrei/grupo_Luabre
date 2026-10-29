#!/usr/bin/env python3
"""Prueba rápida: ¿el pool de 200 elegido por B0 + texto contiene más ganadoras que el pool elegido solo por B0?

No entrena nada (no usa LightGBM). Para las licitaciones de validación (t >= corte), arma dos pools por licitación:

    pool B0      las K candidatas con mayor p_b0
    pool mezcla  las K candidatas con mayor norm(p_b0) + 1.5 * norm(sim),  sim = norm(TF-IDF) + 0.5 * norm(MiniLM)

y mide, para K = 100 y 200:
    Hit@pool     fracción de licitaciones con al menos una ganadora dentro del pool
    Recall@pool  fracción de ganadoras que quedan dentro del pool

Es la cota superior de lo que puede ganar el ranker al cambiar el pool: si no entran más ganadoras, no hay nada que ganar.
Sin fuga de información: igual que en los experimentos, el vocabulario de TF-IDF y los perfiles de embeddings de cada
semestre se arman solo con contratos anteriores al inicio de ese semestre.

Uso (desde la raíz del repo, junto a b0.py, experimento.py y modelo_ranker_texto.py):

    python prueba_pool.py --datos data/rd data/d data/it data/r data/j --cache cache
    python prueba_pool.py --datos data/rd --sin-emb          # solo TF-IDF, no necesita vectores ni GPU
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

import b0
from experimento import nuevo_tfidf

W_EMB, ALPHA = 0.5, 1.5
KS = (100, 200)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def norm(x):
    m = x.max()
    return x / m if m > 0 else x


def semestre(t):
    t = pd.Timestamp(t)
    return f"{t.year}S{1 if t.month <= 6 else 2}"


def inicio(s):
    a, n = s.split("S")
    return pd.Timestamp(f"{a}-{'01' if n == '1' else '07'}-01")


def top_k(score, k):
    # desempate estable (por posición); el ranker del grupo usa un hash, la diferencia es mínima
    return set(np.argsort(-score, kind="stable")[:k].tolist())


def vertical(vdir, cache, corte, sin_emb):
    V = pathlib.Path(vdir).name
    lic = pd.read_parquet(f"{vdir}/licitaciones.parquet")
    gan = pd.read_parquet(f"{vdir}/ganadores.parquet")
    emp = pd.read_parquet(f"{vdir}/empresas.parquet")
    hist = pd.read_parquet(f"{vdir}/historial.parquet")
    lic["id"] = lic["id"].astype(str)
    gan["id"] = gan["id"].astype(str)
    val = lic[lic["t"] >= corte].copy()
    ganadoras = gan.groupby("id")["parent_uei"].apply(set).to_dict()
    prior = b0.B0(hist.copy())
    fechas = pd.to_datetime(hist["fecha"])

    if not sin_emb:
        import modelo_ranker_texto as mt
        emb_hist = np.load(pathlib.Path(cache) / f"emb_hist_{V}.npy").astype("float32", copy=False)
        if len(emb_hist) != len(hist):
            raise SystemExit(f"{V}: la caché tiene {len(emb_hist)} vectores y el historial {len(hist)} contratos")
        emb_lic = mt.vectores_licitaciones(V, lic, pathlib.Path(cache))

    cuentas = {(m, k): [0, 0, 0, 0] for m in ("B0", "mezcla") for k in KS}   # lics con hit, lics, ganadoras dentro, ganadoras
    for s in sorted(val["t"].map(semestre).unique()):
        t0 = time.time()
        antes = (fechas < inicio(s)).to_numpy()
        tf = nuevo_tfidf().fit(hist.loc[antes, "descripcion"].fillna(""))
        cand = emp[emp["semestre"] == s].drop_duplicates("parent_uei").reset_index(drop=True)
        ids = cand["parent_uei"].to_numpy()
        M = tf.transform(cand["texto_awards"].fillna(""))
        if not sin_emb:
            m = antes & hist["parent_uei"].notna().to_numpy()
            cod, uni = pd.factorize(hist.loc[m, "parent_uei"].to_numpy())
            suma = np.zeros((len(uni), emb_hist.shape[1]), dtype="float32")
            np.add.at(suma, cod, emb_hist[m])
            nr = np.linalg.norm(suma, axis=1, keepdims=True)
            nr[nr == 0] = 1
            pos = pd.Series(np.arange(len(uni)), index=uni).reindex(ids)
            P = np.zeros((len(ids), suma.shape[1]), dtype="float32")
            ok = pos.notna().to_numpy()
            P[ok] = (suma / nr)[pos[ok].astype(int).to_numpy()]
        for _, q in val[val["t"].map(semestre) == s].iterrows():
            gs = ganadoras.get(q["id"], set())
            if not gs:
                continue
            p = prior.p_b0(q, cand).reindex(ids).fillna(0).to_numpy()
            texto = ("" if pd.isna(q["title"]) else str(q["title"])) + " " + ("" if pd.isna(q["description"]) else str(q["description"]))
            sim = (M @ tf.transform([texto]).T).toarray().ravel()
            sim = norm(sim)
            if not sin_emb:
                sim = norm(sim + W_EMB * norm(np.clip(P @ emb_lic[q["id"]], 0, None)))
            score = norm(p) + ALPHA * sim
            es_g = np.isin(ids, list(gs))
            n_g = int(es_g.sum())
            if n_g == 0:
                continue
            for k in KS:
                for nombre, sc in (("B0", p), ("mezcla", score)):
                    dentro = int(es_g[list(top_k(sc, k))].sum())
                    c = cuentas[(nombre, k)]
                    c[0] += dentro > 0
                    c[1] += 1
                    c[2] += dentro
                    c[3] += n_g
        log(f"  {V} {s}: {len(cand)} candidatas | {time.time() - t0:.0f} s")
    return V, cuentas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", nargs="+", required=True)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--corte", default="2024-01-01")
    ap.add_argument("--sin-emb", action="store_true", help="solo TF-IDF (no usa vectores)")
    a = ap.parse_args()

    total = {(m, k): [0, 0, 0, 0] for m in ("B0", "mezcla") for k in KS}
    filas = []
    for vd in a.datos:
        V, cu = vertical(vd, a.cache, a.corte, a.sin_emb)
        for (m, k), c in cu.items():
            filas.append({"vertical": V, "pool": f"top-{k}", "método": m,
                          "licitaciones": c[1], "Hit@pool": c[0] / max(c[1], 1), "Recall@pool": c[2] / max(c[3], 1)})
            for i in range(4):
                total[(m, k)][i] += c[i]
    for (m, k), c in total.items():
        filas.append({"vertical": "juntas", "pool": f"top-{k}", "método": m,
                      "licitaciones": c[1], "Hit@pool": c[0] / max(c[1], 1), "Recall@pool": c[2] / max(c[3], 1)})
    df = pd.DataFrame(filas)
    print("\nGanadoras dentro del pool, licitaciones de validación (t >= %s):" % a.corte)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        for k in KS:
            sub = df[df["pool"] == f"top-{k}"]
            t = sub.pivot(index="vertical", columns="método", values=["Hit@pool", "Recall@pool"])
            t["N lic"] = sub[sub["método"] == "B0"].set_index("vertical")["licitaciones"]
            print(f"\n--- pool de {k} ---")
            print(t.reindex([pathlib.Path(v).name for v in a.datos] + ["juntas"]))
    print("\nLectura: si 'mezcla' no supera a 'B0' en Hit@pool y Recall@pool del top-200, no tiene sentido cambiar el pool.")


if __name__ == "__main__":
    main()
