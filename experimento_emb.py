"""Experimento con embeddings por vertical (necesita GPU).

Repite, para cada vertical, lo que hace `experimento.py` y le suma embeddings de texto:

* un vector por contrato del historial (modelo all-MiniLM-L6-v2, hasta 256 tokens);
* el perfil de cada empresa es el promedio de los vectores de sus contratos anteriores al
  comienzo del semestre de la licitación (con vector de largo 1);
* la similitud es el coseno entre el perfil y el vector de la licitación (los negativos se
  recortan a 0).

Compara tres variantes sobre B0, con la misma grilla de alphas:

    tfidf   B0 + TF-IDF (la base, igual que en experimento.py)
    emb     B0 + embeddings
    mezcla  B0 + (TF-IDF + 0.5 * embeddings), cada similitud normalizada por su máximo

Elige la mejor combinación (variante, alpha) de emb y mezcla en selección (2023) y la mide en
prueba (2024S1) contra B0 y contra la base TF-IDF, con IC95 por bootstrap pareado.

Uso:

    python experimento_emb.py rd
    python experimento_emb.py d it r j --salida resultados

Escribe, en `--salida`: resumen_emb_<V>.csv y por_licitacion_emb_<V>.csv. Los vectores de los
contratos se guardan en `--cache` para no recalcularlos.
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
from scipy.sparse import csr_matrix

import b0
import evaluar as ev
import experimento as base

W_MEZCLA = 0.5


def cargar_modelo(nombre: str = "all-MiniLM-L6-v2", largo: int = 256):
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(nombre, device="cuda")
    modelo.max_seq_length = largo
    modelo.half()
    return modelo


def codificar(modelo, textos, lote: int = 128) -> np.ndarray:
    return modelo.encode(list(textos), batch_size=lote, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True).astype("float32")


def puntuar_emb(lics, sc_tfidf, emp, hist, emb_hist, emb_lic, fila_lic) -> dict:
    """Misma forma que `experimento.puntuar`: (ids, prior de B0, similitud). Reusa el prior."""
    sc = {}
    fechas = pd.to_datetime(hist["fecha"])
    for s in sorted(lics["t"].map(base.semestre_de).unique()):
        t0 = time.time()
        corte = base.inicio_semestre(s)
        m = ((fechas < corte) & hist["parent_uei"].notna()).to_numpy()
        codigos, unicos = pd.factorize(hist.loc[m, "parent_uei"].to_numpy())
        n = len(codigos)
        suma = csr_matrix((np.ones(n, dtype="float32"), (codigos, np.arange(n))),
                          shape=(len(unicos), n)) @ emb_hist[m]
        norma = np.linalg.norm(suma, axis=1, keepdims=True)
        norma[norma == 0] = 1
        perfil = suma / norma

        cand = emp[emp.semestre == s].reset_index(drop=True)
        ids = cand["parent_uei"].to_numpy()
        pos = pd.Series(np.arange(len(unicos)), index=unicos).reindex(ids)
        tiene = pos.notna().to_numpy()
        P = np.zeros((len(ids), perfil.shape[1]), dtype="float32")
        P[tiene] = perfil[pos[tiene].astype(int).to_numpy()]

        de_s = lics[lics["t"].map(base.semestre_de) == s]
        for _, q in de_s.iterrows():
            ids_t, p, _sim_t = sc_tfidf[q["id"]]
            v = emb_lic[fila_lic[q["id"]]]
            sc[q["id"]] = (ids_t, p, np.clip(P @ v, 0, None))
        print(f"  {s}: perfiles de {len(unicos)} empresas con historial | licitaciones: {len(de_s)} "
              f"| candidatas: {len(cand)} | {time.time() - t0:.0f} s", flush=True)
    return sc


def mezclar(sc_a: dict, sc_b: dict, w: float) -> dict:
    return {l: (sc_a[l][0], sc_a[l][1], base.norm(sc_a[l][2]) + w * base.norm(sc_b[l][2]))
            for l in sc_a}


def mrr_por_lic(sc, alpha, g) -> pd.Series:
    rk = base.ranking(sc, alpha)
    rk = rk[rk.id.isin(set(g.id))]
    return ev.por_licitacion(rk, g)["MRR"]


def comparar(sc_n, a_n, sc_r, a_r, lics, gan) -> dict:
    g = gan[gan.id.isin(lics.id)]
    a, b = mrr_por_lic(sc_n, a_n, g).align(mrr_por_lic(sc_r, a_r, g), join="inner")
    lo, hi = ev.bootstrap_delta(a.values, b.values)
    return {"Δ MRR": a.mean() - b.mean(), "IC95 bajo": lo, "IC95 alto": hi,
            "mejoran": int((a > b).sum()), "empeoran": int((a < b).sum()), "iguales": int((a == b).sum())}


def correr_vertical(V, a, modelo) -> None:
    t_ini = time.time()
    d = pathlib.Path(a.datos) / V
    print(f"\n=== Vertical {V} (embeddings) ===", flush=True)
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
    print(f"selección: {len(sel)} licitaciones | prueba: {len(pru)} licitaciones | "
          f"contratos en el historial: {len(hist)}", flush=True)

    # 1) vectores de los contratos (con caché) y de las licitaciones
    cache = pathlib.Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    suf = "" if (a.modelo, a.largo) == ("all-MiniLM-L6-v2", 256) else f"_{a.modelo.replace('/', '_')}_{a.largo}"
    ruta_emb = cache / f"emb_hist_{V}{suf}.npy"
    emb_hist = np.load(ruta_emb) if ruta_emb.exists() else None
    if emb_hist is None or len(emb_hist) != len(hist):
        t0 = time.time()
        emb_hist = codificar(modelo, hist["descripcion"].fillna(""))
        np.save(ruta_emb, emb_hist)
        print(f"Vectores de {len(hist)} contratos: {time.time() - t0:.0f} s", flush=True)
    else:
        print(f"Vectores de contratos leídos de {ruta_emb}", flush=True)
    emb_lic = codificar(modelo, lic["title"].map(base.txt) + " " + lic["description"].map(base.txt))
    fila_lic = dict(zip(lic["id"], range(len(lic))))

    # 2) puntajes: TF-IDF (base) y embeddings, y la mezcla
    prior = b0.B0(hist.copy())
    var = {}
    for etapa, lics in (("seleccion", sel), ("prueba", pru)):
        print(f"Puntajes de {etapa}:", flush=True)
        sc_t = base.puntuar(lics, emp, hist, prior)
        sc_e = puntuar_emb(lics, sc_t, emp, hist, emb_hist, emb_lic, fila_lic)
        var[etapa] = {"tfidf": sc_t, "emb": sc_e, "mezcla": mezclar(sc_t, sc_e, W_MEZCLA)}

    # 3) medir todas las variantes y alphas en las dos etapas
    filas, por_lic, mrr_sel = [], [], {}
    for etapa, lics in (("seleccion", sel), ("prueba", pru)):
        g = gan[gan.id.isin(lics.id)]
        pl0 = ev.por_licitacion(rk_b0[rk_b0.id.isin(set(g.id))], g)
        por_lic.append(pl0.assign(vertical=V, etapa=etapa, variante="B0", alpha="B0").reset_index())
        for nombre, sc in var[etapa].items():
            for al in a.alphas:
                res, pl = base.medir(sc, al, lics, lic, emp, gan, rk_b0)
                filas.append(res.assign(vertical=V, etapa=etapa, variante=nombre, alpha=al))
                por_lic.append(pl.assign(vertical=V, etapa=etapa, variante=nombre, alpha=al).reset_index())
                if etapa == "seleccion":
                    mrr_sel[(nombre, al)] = float(res.loc[res.conjunto == "todas", "MRR"].iloc[0])
        print(f"Etapa {etapa} medida ({time.time() - t_ini:.0f} s desde el inicio)", flush=True)

    resumen = pd.concat(filas, ignore_index=True)
    sal = pathlib.Path(a.salida)
    sal.mkdir(parents=True, exist_ok=True)
    resumen.to_csv(sal / f"resumen_emb{a.etiqueta}_{V}.csv", index=False)
    pd.concat(por_lic, ignore_index=True).to_csv(sal / f"por_licitacion_emb{a.etiqueta}_{V}.csv", index=False)

    # 4) tablas y decisión
    tabla = pd.DataFrame({n: [mrr_sel[(n, al)] for al in a.alphas] for n in ("tfidf", "emb", "mezcla")},
                         index=pd.Index(a.alphas, name="alpha"))
    a_t = max(a.alphas, key=lambda al: mrr_sel[("tfidf", al)])
    nombre, a_n = max(((n, al) for n in ("emb", "mezcla") for al in a.alphas), key=lambda x: mrr_sel[x])
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print("\nMRR en selección (2023), todas las licitaciones:")
        print(tabla.to_string())
        print(f"\nBase TF-IDF: alpha {a_t} | mejor con embeddings: {nombre}, alpha {a_n}")
        for etapa, lics in (("seleccion", sel), ("prueba", pru)):
            c = comparar(var[etapa][nombre], a_n, var[etapa]["tfidf"], a_t, lics, gan)
            print(f"{etapa:9s}: {nombre} (alpha {a_n}) contra la base TF-IDF (alpha {a_t}): "
                  f"Δ MRR {c['Δ MRR']:+.3f}, IC95 ({c['IC95 bajo']:+.3f}, {c['IC95 alto']:+.3f}) | "
                  f"mejoran {c['mejoran']}, empeoran {c['empeoran']}, iguales {c['iguales']}")
        r = resumen[(resumen.etapa == "prueba")
                    & (((resumen.variante == "tfidf") & (resumen.alpha == a_t))
                       | ((resumen.variante == nombre) & (resumen.alpha == a_n)))]
        print("\nPrueba (2024S1) contra B0, por conjunto:")
        print(r[["variante", "alpha", "conjunto", "N", "MRR", "B0 MRR", "Δ MRR", "Δ MRR IC95", "Hit@5", "B0 Hit@5"]]
              .to_string(index=False))
    print(f"\nTiempo total {V}: {time.time() - t_ini:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("verticales", nargs="+", help="rd, d, it, r y/o j")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--b0dir", default=".")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--modelo", default="all-MiniLM-L6-v2", help="modelo de sentence-transformers")
    ap.add_argument("--largo", type=int, default=256, help="máximo de tokens que lee el modelo")
    ap.add_argument("--etiqueta", default="", help="sufijo para la caché y los archivos de salida, p. ej. _mpnet")
    ap.add_argument("--alphas", default=",".join(str(x) for x in base.ALPHAS))
    a = ap.parse_args()
    a.alphas = [float(x) for x in a.alphas.split(",")]
    print(f"Modelo: {a.modelo} (largo {a.largo}) | alphas: {a.alphas}", flush=True)
    modelo = cargar_modelo(a.modelo, a.largo)
    for V in a.verticales:
        try:
            correr_vertical(V, a, modelo)
        except Exception:
            print(f"\n*** Falló la vertical {V}:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    main()
