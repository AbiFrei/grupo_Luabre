"""Experimento por vertical: B0 + TF-IDF con validación cronológica.

Para una vertical `V` (rd, d, it, r, j), hace lo mismo que las celdas del notebook de
exploración (Experimentación i):

1. Separa las licitaciones en selección (2023) y prueba (desde 2024-01-01).
2. Para cada licitación calcula, usando solo contratos anteriores al comienzo de su
   semestre, el prior de B0 y la similitud TF-IDF entre la licitación y cada empresa candidata.
3. Arma el ranking `norm(p) + alpha * norm(sim)` para una grilla de alphas.
4. Elige el alpha con mejor MRR en selección y lo mide en prueba contra B0 (con IC95).

Uso:

    python experimento.py rd
    python experimento.py j --datos data --salida resultados

Si falta `b0_<V>.csv` (el ranking de B0), lo genera con `b0.py`.

Escribe, en la carpeta `--salida`:

    resumen_<V>.csv           una fila por (etapa, alpha, conjunto) con las métricas y el Δ contra B0
    por_licitacion_<V>.csv    MRR, Hit@5 y Hit@1 de cada licitación, para cada alpha y para B0
                              (sirve para comparar un alpha global con uno por vertical)
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import b0
import evaluar as ev

ALPHAS = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]
SEL_DESDE, PRUEBA_DESDE = "2023-01-01", "2024-01-01"


def semestre_de(t) -> str:
    t = pd.Timestamp(t)
    return f"{t.year}S{1 if t.month <= 6 else 2}"


def inicio_semestre(s: str) -> str:
    anio, n = s.split("S")
    return f"{anio}-{'01' if n == '1' else '07'}-01"


def txt(x) -> str:
    return "" if pd.isna(x) else str(x)


def norm(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x / m if m > 0 else x


def nuevo_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                           min_df=2, max_features=100_000, sublinear_tf=True)


def puntuar(lics, emp, hist, prior) -> dict:
    """Por licitación: (ids de candidatas, prior de B0, similitud TF-IDF). Sin mirar el futuro."""
    sc = {}
    for s in sorted(lics["t"].map(semestre_de).unique()):
        t0 = time.time()
        corte = inicio_semestre(s)
        textos = hist.loc[pd.to_datetime(hist.fecha) < corte, "descripcion"].fillna("")
        tf = nuevo_tfidf().fit(textos)
        cand = emp[emp.semestre == s].reset_index(drop=True)
        M = tf.transform(cand["texto_awards"].fillna(""))
        ids = cand["parent_uei"].to_numpy()
        de_s = lics[lics["t"].map(semestre_de) == s]
        for _, q in de_s.iterrows():
            p = prior.p_b0(q, cand).reindex(cand["parent_uei"]).fillna(0).to_numpy()
            v = tf.transform([txt(q["title"]) + " " + txt(q["description"])])
            sc[q["id"]] = (ids, p, (M @ v.T).toarray().ravel())
        print(f"  {s}: vocabulario con contratos antes de {corte} | licitaciones: {len(de_s)} "
              f"| candidatas: {len(cand)} | {time.time() - t0:.0f} s", flush=True)
    return sc


def ranking(sc: dict, alpha: float, top: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    filas = []
    for lid, (ids, p, sim) in sc.items():
        s = norm(p) + alpha * norm(sim) + 1e-9 * rng.random(len(p))
        for puesto, i in enumerate(np.argsort(-s)[:top], 1):
            filas.append((lid, ids[i], puesto))
    return pd.DataFrame(filas, columns=["id", "parent_uei", "puesto"])


def medir(sc, alpha, lics, lic, emp, gan, rk_b0):
    """Devuelve (resumen por conjunto, MRR por licitación) del ranking con ese alpha."""
    g = gan[gan.id.isin(lics.id)]
    rk = ranking(sc, alpha)
    rk = rk[rk.id.isin(set(g.id))]
    rk0 = rk_b0[rk_b0.id.isin(set(g.id))]
    res = ev.evaluar(rk, lic, emp, g, rk0)
    return res, ev.por_licitacion(rk, g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("vertical", help="rd, d, it, r o j")
    ap.add_argument("--datos", default="data", help="carpeta con una subcarpeta por vertical")
    ap.add_argument("--b0dir", default=".", help="dónde está (o se escribe) b0_<V>.csv")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    a = ap.parse_args()

    V = a.vertical
    alphas = [float(x) for x in a.alphas.split(",")]
    d = pathlib.Path(a.datos) / V
    t_ini = time.time()
    print(f"=== Vertical {V} ===", flush=True)

    lic = pd.read_parquet(d / "licitaciones.parquet")
    gan = pd.read_parquet(d / "ganadores.parquet")
    emp = pd.read_parquet(d / "empresas.parquet")
    hist = pd.read_parquet(d / "historial.parquet")

    ruta_b0 = pathlib.Path(a.b0dir) / f"b0_{V}.csv"
    if not ruta_b0.exists():
        print(f"Falta {ruta_b0}: lo generamos con b0.py", flush=True)
        subprocess.run([sys.executable, "b0.py", str(d), "--salida", str(ruta_b0)], check=True)
    rk_b0 = pd.read_csv(ruta_b0)

    sel = lic[(lic.t >= SEL_DESDE) & (lic.t < PRUEBA_DESDE)].copy()
    pru = lic[lic.t >= PRUEBA_DESDE].copy()
    for nombre, x in (("selección", sel), ("prueba", pru)):
        print(f"{nombre:10s} | licitaciones: {len(x):4d} | ganadores: {gan.id.isin(x.id).sum():4d} "
              f"| semestres: {sorted(x.t.map(semestre_de).unique())}", flush=True)
    if len(sel) == 0 or len(pru) == 0:
        sys.exit("La selección o la prueba quedaron vacías: revisar las fechas de licitaciones.parquet")

    prior = b0.B0(hist.copy())
    print("Puntajes de selección:", flush=True)
    sc_sel = puntuar(sel, emp, hist, prior)
    print("Puntajes de prueba:", flush=True)
    sc_pru = puntuar(pru, emp, hist, prior)

    filas, por_lic = [], []
    mrr_sel = {}
    for etapa, lics, sc in (("seleccion", sel, sc_sel), ("prueba", pru, sc_pru)):
        g = gan[gan.id.isin(lics.id)]
        pl0 = ev.por_licitacion(rk_b0[rk_b0.id.isin(set(g.id))], g)
        por_lic.append(pl0.assign(vertical=V, etapa=etapa, alpha="B0").reset_index())
        for al in alphas:
            res, pl = medir(sc, al, lics, lic, emp, gan, rk_b0)
            res = res.assign(vertical=V, etapa=etapa, alpha=al)
            filas.append(res)
            por_lic.append(pl.assign(vertical=V, etapa=etapa, alpha=al).reset_index())
            if etapa == "seleccion":
                mrr_sel[al] = float(res.loc[res.conjunto == "todas", "MRR"].iloc[0])
        print(f"Etapa {etapa} medida ({time.time() - t_ini:.0f} s desde el inicio)", flush=True)

    resumen = pd.concat(filas, ignore_index=True)
    sal = pathlib.Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(sal / f"resumen_{V}.csv", index=False)
    pd.concat(por_lic, ignore_index=True).to_csv(sal / f"por_licitacion_{V}.csv", index=False)

    fmt = {"display.float_format": "{:.3f}".format, "display.width": 220, "display.max_columns": None}
    cols = ["alpha", "N", "MRR", "Hit@5", "Hit@1", "Δ MRR", "Δ MRR IC95"]
    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        t = resumen[(resumen.etapa == "seleccion") & (resumen.conjunto == "todas")]
        print("\nSelección (2023), todas las licitaciones:")
        print(t[cols].to_string(index=False))
        elegido = max(mrr_sel, key=mrr_sel.get)
        print(f"\nalpha elegido en selección: {elegido}")
        t = resumen[(resumen.etapa == "prueba") & (resumen.alpha == elegido)]
        print(f"\nPrueba (2024S1) con alpha = {elegido}, por conjunto:")
        print(t[["conjunto", "N", "MRR", "B0 MRR", "Δ MRR", "Δ MRR IC95", "Hit@5", "B0 Hit@5"]]
              .to_string(index=False))
    print(f"\nTiempo total: {time.time() - t_ini:.0f} s | archivos en {sal}/", flush=True)


if __name__ == "__main__":
    main()
