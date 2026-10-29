#!/usr/bin/env python3
"""Ranker del historial (LightGBM) + features de texto (TF-IDF y embeddings MiniLM).

Usa `modelo_ranker.py` del grupo SIN modificarlo: lo importamos y le agregamos, a cada par
(licitación, empresa del pool), las features de texto de nuestra mezcla:

    sim_tfidf        coseno entre la licitación y el texto de los contratos previos de la empresa (TF-IDF)
    sim_emb          coseno entre la licitación y el perfil de embeddings MiniLM de la empresa
    *_rel            cada similitud dividida por su máximo en el pool de la licitación
    sim_mezcla       sim_tfidf_rel + 0.5 * sim_emb_rel
    score_mezcla     p_rel + 1.5 * sim_mezcla   (nuestra fórmula, con el alpha global 1.5)
    rank_*           posición dentro del pool según cada similitud / según score_mezcla

Sin fuga de información: para cada semestre, el vocabulario de TF-IDF y los perfiles de embeddings
se arman solo con contratos anteriores al inicio de ese semestre (igual que en nuestros experimentos).

Uso (desde la raíz del repo, junto a b0.py, evaluar.py y modelo_ranker.py):

    python modelo_ranker_texto.py --datos data/rd
    python modelo_ranker_texto.py --todas --salida-dir rankings_texto
    python modelo_ranker_texto.py --todas --sin-texto --salida-dir rankings_sin_texto

Los vectores MiniLM de los contratos se guardan en `--cache` (por defecto `cache/`). Si no están, se calculan
desde `historial.parquet` la primera vez (con GPU tarda unos minutos por vertical; con CPU, horas).

`--sin-texto` entrena el mismo modelo sin las features nuevas (debería reproducir los rankings del grupo).
Cómo se elige el pool de `--pool` candidatas (200) que reordena el LightGBM (la constante POOL_POR_TEXTO_DEFECTO decide cuál
es el modo por defecto; las dos banderas lo cambian en esa corrida):
    `--pool-por-texto`  las 200 con B0 + texto: se ordenan TODAS las candidatas con norm(p_b0) + 1.5 * norm(sim)
    `--pool-por-b0`     las 200 con mejor B0 (el modo anterior, solo historial)
Para volver al modo anterior alcanza con agregar `--pool-por-b0` o con poner POOL_POR_TEXTO_DEFECTO = False.
Modo validar (por defecto): entrena con t < corte y valida con t >= corte.
Modo final: entrena con todas las licitaciones y rankea las del test:

    python modelo_ranker_texto.py --todas --modo final --test-raiz <raiz_test> --salida-dir entrega

En modo final escribe un CSV por vertical con el nombre que pide la entrega (rd.csv, d.csv, it.csv, r.csv, j.csv).
Si `empresas.parquet` del test no trae la columna `semestre` (una sola foto de empresas), todas las filas son candidatas
de todas las licitaciones y los vocabularios / perfiles de texto se arman con los contratos anteriores a la primera
licitación del test.

<raiz_test>/<vertical>/ debe tener licitaciones.parquet y empresas.parquet (historial.parquet es opcional:
si falta se usa el de entrenamiento). Los contratos nuevos del historial del test se codifican al vuelo.
"""
from __future__ import annotations

import argparse
import shutil
import pathlib
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import modelo_ranker as mr
from experimento import nuevo_tfidf     # el mismo TF-IDF de nuestros experimentos (corte determinista)

W_EMB = 0.5          # peso de los embeddings dentro de la similitud (el mismo de nuestra mezcla)
ALPHA_REF = 1.5      # alpha global de nuestra mezcla
POOL_POR_TEXTO_DEFECTO = True    # True: el pool de 200 se elige con B0 + texto; False: solo con B0 (modo anterior). Se cambia por corrida con --pool-por-texto / --pool-por-b0
LOTE_LIC = 64        # licitaciones por bloque al calcular similitudes
ESTADO: dict = {}    # datos de la vertical que se está procesando (entrenamiento / validación)
ESTADO_TEST: dict = {}   # datos del test de la vertical (solo en modo final con --test-raiz)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def txt(x) -> str:
    return "" if pd.isna(x) else str(x)


def inicio_semestre(s: str) -> pd.Timestamp:
    anio, n = s.split("S")
    return pd.Timestamp(f"{anio}-{'01' if n == '1' else '07'}-01")


def corte_de(E: dict, s: str) -> pd.Timestamp:
    """Fecha desde la cual no se usa información: inicio del semestre `s`. Para '*' (empresas sin columna `semestre`, como en el
    test) es la primera licitación del test; si no se conoce, el día siguiente al último contrato del historial."""
    if s != "*":
        return inicio_semestre(s)
    if E.get("t_min") is not None:
        return pd.Timestamp(E["t_min"])
    return pd.to_datetime(E["hist"]["fecha"]).max() + pd.Timedelta(days=1)


def candidatas_de(emp: pd.DataFrame, s: str) -> pd.DataFrame:
    """Candidatas del semestre `s`. Sin columna `semestre` (el test trae una sola foto) o con s == '*': todas las filas."""
    if s == "*" or "semestre" not in emp.columns:
        return emp.drop_duplicates("parent_uei").reset_index(drop=True)
    return emp[emp["semestre"] == s].drop_duplicates("parent_uei").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Vectores de las licitaciones (los de los contratos vienen de la caché)
# --------------------------------------------------------------------------- #
_MODELO: dict = {}


def _modelo():
    """MiniLM como en experimento_emb.py (en GPU, media precisión, hasta 256 tokens). Se carga una sola vez."""
    if "m" not in _MODELO:
        import torch
        from sentence_transformers import SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            log("ADVERTENCIA: no hay GPU; codificar los contratos del historial puede tardar horas.")
        m = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
        m.max_seq_length = 256
        if dev == "cuda":
            m.half()
        _MODELO["m"] = m
    return _MODELO["m"]


def codificar_licitaciones(textos: list[str], barra: bool = False) -> np.ndarray:
    return _modelo().encode(list(textos), batch_size=128, normalize_embeddings=True,
                            show_progress_bar=barra, convert_to_numpy=True).astype("float32")


def vectores_licitaciones(V: str, lic: pd.DataFrame, cache: pathlib.Path) -> dict:
    """{id: vector}. Se guardan en la caché; solo se codifican las licitaciones que faltan."""
    ruta = cache / f"emb_lic_{V}.npz"
    ids, vecs = np.array([], dtype=str), np.zeros((0, 384), dtype="float32")
    if ruta.exists():
        z = np.load(ruta, allow_pickle=False)
        ids, vecs = z["ids"], z["vecs"]
    conocidos = set(ids.tolist())
    faltan = lic[~lic["id"].astype(str).isin(conocidos)]
    if len(faltan):
        t0 = time.time()
        nuevos = codificar_licitaciones((faltan["title"].map(txt) + " " + faltan["description"].map(txt)).tolist())
        ids = np.concatenate([ids, np.asarray(faltan["id"].astype(str).tolist(), dtype=str)])
        vecs = np.concatenate([vecs, nuevos])
        np.savez(ruta, ids=ids, vecs=vecs)
        log(f"{V}: {len(faltan)} licitaciones codificadas en {time.time() - t0:.0f} s")
    return dict(zip(ids.tolist(), vecs))


# --------------------------------------------------------------------------- #
# Features de texto para las filas (licitación x empresa) de un DataFrame de `construir`
# --------------------------------------------------------------------------- #
def agregar_texto(df: pd.DataFrame, E: dict) -> pd.DataFrame:
    hist, emp, lic_txt, emb_hist, emb_lic = E["hist"], E["emp"], E["lic_txt"], E["emb_hist"], E["emb_lic"]
    fechas = pd.to_datetime(hist["fecha"])
    con_uei = hist["parent_uei"].notna().to_numpy()
    sim_t = np.zeros(len(df), dtype="float32")
    sim_e = np.zeros(len(df), dtype="float32")
    ids_df = df["id"].astype(str).to_numpy()
    uei_df = df["parent_uei"].astype(str).to_numpy()

    for s, filas in df.groupby("_sem").indices.items():
        t0 = time.time()
        corte = corte_de(E, s)
        antes = (fechas < corte).to_numpy()
        tf = nuevo_tfidf().fit(hist.loc[antes, "descripcion"].fillna(""))
        cand = candidatas_de(emp, s)
        M = tf.transform(cand["texto_awards"].fillna(""))

        m = antes & con_uei                                       # perfiles de embeddings: promedio de vectores
        codigos, unicos = pd.factorize(hist.loc[m, "parent_uei"].to_numpy())
        n = len(codigos)
        suma = csr_matrix((np.ones(n, dtype="float32"), (codigos, np.arange(n))), shape=(len(unicos), n)) @ emb_hist[m]
        norma = np.linalg.norm(suma, axis=1, keepdims=True)
        norma[norma == 0] = 1
        perfil = suma / norma
        P = np.zeros((len(cand), perfil.shape[1]), dtype="float32")
        pos_u = pd.Series(np.arange(len(unicos)), index=unicos).reindex(cand["parent_uei"].to_numpy())
        tiene = pos_u.notna().to_numpy()
        P[tiene] = perfil[pos_u[tiene].astype(int).to_numpy()]

        pos_c = pd.Series(np.arange(len(cand)), index=cand["parent_uei"].astype(str).to_numpy())
        ci = pos_c.reindex(uei_df[filas]).to_numpy()
        ok = ~np.isnan(ci)
        ci = np.where(ok, ci, 0).astype(int)

        lics = pd.unique(ids_df[filas])
        for i in range(0, len(lics), LOTE_LIC):
            bloque = lics[i:i + LOTE_LIC]
            St = (tf.transform([lic_txt[x] for x in bloque]) @ M.T).toarray()
            Se = np.clip(np.stack([emb_lic[x] for x in bloque]) @ P.T, 0, None)
            li = pd.Series(np.arange(len(bloque)), index=bloque).reindex(ids_df[filas]).to_numpy()
            r = np.flatnonzero(~np.isnan(li) & ok)
            lr = li[r].astype(int)
            sim_t[filas[r]] = St[lr, ci[r]]
            sim_e[filas[r]] = Se[lr, ci[r]]
        log(f"  texto {s}: {len(lics)} licitaciones, {len(cand)} candidatas, {len(filas):,} pares | {time.time() - t0:.0f} s")

    df = df.copy()
    df["sim_tfidf"], df["sim_emb"] = sim_t, sim_e
    g = df.groupby("lic_pos")
    mt, me = g["sim_tfidf"].transform("max").to_numpy(), g["sim_emb"].transform("max").to_numpy()
    df["sim_tfidf_rel"] = np.where(mt > 0, df["sim_tfidf"] / np.where(mt > 0, mt, 1), 0).astype("float32")
    df["sim_emb_rel"] = np.where(me > 0, df["sim_emb"] / np.where(me > 0, me, 1), 0).astype("float32")
    df["sim_mezcla"] = df["sim_tfidf_rel"] + W_EMB * df["sim_emb_rel"]
    df["score_mezcla"] = df["p_rel"] + ALPHA_REF * df["sim_mezcla"]
    g = df.groupby("lic_pos")
    for c in ("sim_tfidf", "sim_emb", "score_mezcla"):
        df[f"rank_{c}"] = g[c].rank(ascending=False, method="first").astype("float32")
    return df


# --------------------------------------------------------------------------- #
# Modo final: el test trae sus licitaciones, sus empresas y, opcionalmente, un historial propio
# --------------------------------------------------------------------------- #
def clave_contrato(h: pd.DataFrame) -> np.ndarray:
    """Huella de cada contrato (empresa, fecha, descripción) para reusar los vectores de la caché."""
    f = pd.to_datetime(h["fecha"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    s = h["parent_uei"].astype(str) + "|" + f + "|" + h["descripcion"].fillna("").astype(str)
    return pd.util.hash_pandas_object(s, index=False).to_numpy()


def alinear_embeddings(E: dict, hist: pd.DataFrame) -> np.ndarray:
    """Vectores de un historial nuevo: los contratos ya conocidos salen de la caché; los nuevos se codifican."""
    k_tr, k_te = clave_contrato(E["hist"]), clave_contrato(hist)
    pos = pd.Series(np.arange(len(k_tr)), index=k_tr)
    pos = pos[~pos.index.duplicated()]
    m = pos.reindex(k_te).to_numpy()
    ok = ~np.isnan(m)
    out = np.zeros((len(hist), E["emb_hist"].shape[1]), dtype="float32")
    out[ok] = E["emb_hist"][m[ok].astype(int)]
    nuevos = np.flatnonzero(~ok)
    if len(nuevos):
        log(f"historial del test: {len(nuevos):,} contratos nuevos para codificar de {len(hist):,}")
        out[nuevos] = codificar_licitaciones(hist["descripcion"].fillna("").astype(str).iloc[nuevos].tolist())
    return out


def preparar_test(lic_t: pd.DataFrame, emp_t: pd.DataFrame, hist_t) -> None:
    """Deja en ESTADO_TEST lo que necesitan las features de texto para las licitaciones del test."""
    E = ESTADO
    if hist_t is None:                       # sin historial propio: el del entrenamiento (y su caché)
        hist, emb_hist = E["hist"], E["emb_hist"]
    else:
        hist = hist_t.reset_index(drop=True)
        emb_hist = alinear_embeddings(E, hist)
    for c in ("title", "description"):
        if c not in lic_t.columns:
            mr.warn(f"licitaciones del test sin columna '{c}': las features de texto usan solo la otra.")
            lic_t = lic_t.assign(**{c: ""})
    ids = lic_t["id"].astype(str).tolist()
    textos = (lic_t["title"].map(txt) + " " + lic_t["description"].map(txt)).tolist()
    t0 = time.time()
    vecs = codificar_licitaciones(textos) if len(textos) else np.zeros((0, 384), dtype="float32")
    log(f"test: {len(ids):,} licitaciones codificadas en {time.time() - t0:.0f} s")
    ESTADO_TEST.clear()
    ESTADO_TEST.update(hist=hist, emp=emp_t, emb_hist=emb_hist, lic_txt=dict(zip(ids, textos)),
                       emb_lic=dict(zip(ids, vecs)), ids=set(ids), t_min=pd.Timestamp(lic_t["t"].min()))


_cargar_test_original = mr.cargar_test


def cargar_test_con_texto(td):
    lic_t, emp_t, hist_t, gan = _cargar_test_original(td)
    if ESTADO.get("texto", True):
        preparar_test(lic_t, emp_t, hist_t)
    return lic_t, emp_t, hist_t, gan


mr.cargar_test = cargar_test_con_texto   # procesar() busca `cargar_test` en el módulo en cada llamada


_construir_original = mr.construir
OPCIONES = {"pool_texto": False}     # --pool-por-texto
ESTADO_B0: dict = {}                 # id de licitación -> empresas del top del B0 real (solo con --pool-por-texto)


def _fila_max1(x: np.ndarray) -> np.ndarray:
    """Divide cada fila por su máximo (las filas con máximo 0 quedan en 0)."""
    m = x.max(axis=1, keepdims=True)
    return np.divide(x, m, out=np.zeros_like(x), where=m > 0)


def _texto_semestre(E: dict, s: str):
    """TF-IDF y perfiles de embeddings de las candidatas del semestre `s` (solo contratos anteriores al semestre)."""
    cache = E.setdefault("_sem", {})
    if s not in cache:
        t0 = time.time()
        hist, emp, emb_hist = E["hist"], E["emp"], E["emb_hist"]
        if "_fechas" not in E:
            E["_fechas"] = pd.to_datetime(hist["fecha"])
        antes = (E["_fechas"] < corte_de(E, s)).to_numpy()
        tf = nuevo_tfidf().fit(hist.loc[antes, "descripcion"].fillna(""))
        cand = candidatas_de(emp, s)
        M = tf.transform(cand["texto_awards"].fillna(""))
        m = antes & hist["parent_uei"].notna().to_numpy()
        codigos, unicos = pd.factorize(hist.loc[m, "parent_uei"].to_numpy())
        n = len(codigos)
        suma = csr_matrix((np.ones(n, dtype="float32"), (codigos, np.arange(n))), shape=(len(unicos), n)) @ emb_hist[m]
        norma = np.linalg.norm(suma, axis=1, keepdims=True)
        norma[norma == 0] = 1
        perfil = suma / norma
        P = np.zeros((len(cand), perfil.shape[1]), dtype="float32")
        pos_u = pd.Series(np.arange(len(unicos)), index=unicos).reindex(cand["parent_uei"].to_numpy())
        tiene = pos_u.notna().to_numpy()
        P[tiene] = perfil[pos_u[tiene].astype(int).to_numpy()]
        pos_c = pd.Series(np.arange(len(cand)), index=cand["parent_uei"].astype(str).to_numpy())
        cache[s] = (tf, M, P, pos_c)
        log(f"  pool por texto {s}: perfiles de {len(cand)} candidatas | {time.time() - t0:.0f} s")
    return cache[s]


def construir_pool_texto(idx, ctx, ei, pos, pool, universo, gan=None):
    """Igual que mr.construir, pero el pool (top-`pool`) se elige con norm(p_b0) + ALPHA_REF * norm(sim) sobre TODAS
    las candidatas del semestre, y no solo con p_b0. rank_b0 sigue siendo el puesto real del B0 entre todas las
    candidatas, y p_rel sigue siendo p_b0 relativo al máximo de todas las candidatas."""
    E = ESTADO_TEST if (bool(ESTADO_TEST) and gan is None) else ESTADO
    sem = ctx.sem[pos] if ei.tiene_sem else np.full(len(pos), "*")     # sin `semestre` (test): un solo grupo con todas
    bloques, sin_cands = [], 0
    for s in pd.unique(sem):
        lp = pos[sem == s]
        cs = ei.by_sem.get(s)
        if cs is None or len(cs) == 0:
            sin_cands += len(lp)
            continue
        tf, M, P, pos_c = _texto_semestre(E, s)
        pc = pos_c.reindex(universo[cs]).to_numpy()          # lugar de cada candidata en las matrices de texto
        ok = ~np.isnan(pc)
        pci = np.where(ok, pc, 0).astype(int)
        cu = mr.splitmix64(cs.astype(np.uint64))
        paso = max(1, min(mr.MAX_PARES // len(cs), LOTE_LIC))
        K = min(pool, len(cs))
        for i in range(0, len(lp), paso):
            lb = lp[i:i + paso]
            p = idx.p_denso(ctx, lb, cs)
            tie = mr.splitmix64(ctx.seed[lb][:, None] ^ cu[None, :])
            orden_b0 = np.lexsort((tie, -p), axis=1)                     # orden real del B0 (todas las candidatas)
            ids_b = [str(x) for x in ctx.ids[lb]]
            St = (tf.transform([E["lic_txt"][x] for x in ids_b]) @ M.T).toarray()[:, pci] * ok
            Se = np.clip(np.stack([E["emb_lic"][x] for x in ids_b]) @ P.T, 0, None)[:, pci] * ok
            sim = _fila_max1(_fila_max1(St.astype("float64")) + W_EMB * _fila_max1(Se.astype("float64")))
            score = _fila_max1(p) + ALPHA_REF * sim
            orden = np.lexsort((tie, -score), axis=1)[:, :K]             # pool: top-K por B0 + texto
            rb0 = np.empty(orden_b0.shape, dtype=np.int32)
            np.put_along_axis(rb0, orden_b0, np.broadcast_to(np.arange(1, len(cs) + 1, dtype=np.int32), orden_b0.shape), axis=1)
            for j in range(len(lb)):
                ESTADO_B0[ids_b[j]] = universo[cs[orden_b0[j, :mr.TOP]]]
            lsel, csel = np.repeat(lb, K), cs[orden].ravel()
            f = idx.feats_pares(ctx, lsel, csel)
            bloques.append(pd.DataFrame({"lic_pos": lsel, "_c": csel, "_sem": s,
                                         "rank_b0": np.take_along_axis(rb0, orden, axis=1).ravel(),
                                         "_pmax": np.repeat(p.max(axis=1), K), **f}))
    if sin_cands:
        mr.warn(f"{sin_cands} licitaciones sin candidatas en su semestre (no se pueden rankear).")
    if not bloques:
        raise mr.PipelineError("Ninguna licitación tiene candidatas.")
    df = pd.concat(bloques, ignore_index=True)
    df = df.sort_values(["lic_pos", "rank_b0"], kind="stable").reset_index(drop=True)
    df["p_rel"] = (df["p_b0"] / df["_pmax"].replace(0, np.nan)).fillna(0.0)
    df = df.drop(columns="_pmax")
    if len(ei.feat.columns) > 2:
        df = df.merge(ei.feat, on=["_sem", "_c"], how="left")
    df["id"] = ctx.ids[df["lic_pos"].to_numpy()]
    df["parent_uei"] = universo[df["_c"].to_numpy()]
    df["tq"] = ctx.tq[df["lic_pos"].to_numpy()]
    if gan is not None:
        g = gan[["id", "parent_uei"]].drop_duplicates().assign(label=1)
        df = df.merge(g, on=["id", "parent_uei"], how="left")
        df["label"] = df["label"].fillna(0).astype("int8")
    return df


_armar_ranking_original = mr.armar_ranking


def armar_ranking_con_b0_real(df, score, ctx, top=mr.TOP):
    """Con --pool-por-texto, el B0 de referencia de la validación (score = -rank_b0) es el top real del B0 entre todas
    las candidatas, y no el del pool nuevo. El resto de los rankings sale igual que siempre."""
    if OPCIONES["pool_texto"] and ESTADO_B0 and np.array_equal(np.asarray(score), -df["rank_b0"].to_numpy()):
        filas = []
        for lid in pd.unique(df["id"]):
            u = ESTADO_B0[str(lid)][:top]
            filas.append(pd.DataFrame({"id": lid, "parent_uei": u, "puesto": np.arange(1, len(u) + 1)}))
        return pd.concat(filas, ignore_index=True)
    return _armar_ranking_original(df, score, ctx, top)


mr.armar_ranking = armar_ranking_con_b0_real   # procesar() busca `armar_ranking` en el módulo en cada llamada


def construir_con_texto(idx, ctx, ei, pos, pool, universo, gan=None):
    if OPCIONES["pool_texto"] and ESTADO.get("texto", True):
        df = construir_pool_texto(idx, ctx, ei, pos, pool, universo, gan)
    else:
        df = _construir_original(idx, ctx, ei, pos, pool, universo, gan)
    if not ESTADO.get("texto", True):
        return df
    es_test = bool(ESTADO_TEST) and gan is None        # solo las licitaciones del test se construyen sin ganadores
    return agregar_texto(df, ESTADO_TEST if es_test else ESTADO)


mr.construir = construir_con_texto     # procesar() busca `construir` en el módulo en cada llamada


def preparar_vertical(vdir: str, cache: pathlib.Path, texto: bool) -> None:
    V = pathlib.Path(vdir).name
    ESTADO.clear()
    ESTADO_TEST.clear()
    ESTADO_B0.clear()
    ESTADO["texto"] = texto
    if not texto:
        return
    lic = pd.read_parquet(pathlib.Path(vdir) / "licitaciones.parquet")
    emp = pd.read_parquet(pathlib.Path(vdir) / "empresas.parquet")
    hist = pd.read_parquet(pathlib.Path(vdir) / "historial.parquet")      # sin filtrar: alineado con la caché
    ruta = cache / f"emb_hist_{V}.npy"
    if not ruta.exists():
        t0 = time.time()
        log(f"{V}: no hay vectores en {ruta}; se calculan para {len(hist):,} contratos")
        np.save(ruta, codificar_licitaciones(hist["descripcion"].fillna("").astype(str).tolist(), barra=True))
        log(f"{V}: vectores de contratos calculados y guardados en {time.time() - t0:.0f} s")
    emb_hist = np.load(ruta).astype("float32", copy=False)     # acepta vectores guardados en float16 o float32
    if len(emb_hist) != len(hist):
        raise mr.PipelineError(f"{V}: la caché tiene {len(emb_hist)} vectores y el historial {len(hist)} contratos")
    lic["id"] = lic["id"].astype(str)
    ESTADO.update(
        hist=hist.reset_index(drop=True), emp=emp, emb_hist=emb_hist,
        lic_txt=dict(zip(lic["id"], lic["title"].map(txt) + " " + lic["description"].map(txt))),
        emb_lic=vectores_licitaciones(V, lic, cache))
    log(f"{V}: {len(hist):,} contratos con vectores, {len(lic):,} licitaciones, {len(emp):,} filas de empresas")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ranker del historial + features de texto")
    ap.add_argument("--datos", nargs="+", help="carpetas de vertical, p. ej. data/j data/rd")
    ap.add_argument("--todas", action="store_true", help="todas las carpetas data/*/ con licitaciones")
    ap.add_argument("--cache", default="cache", help="carpeta con emb_hist_<V>.npy y emb_lic_<V>.npz; se crea y se llena si no existe")
    ap.add_argument("--sin-texto", action="store_true", help="no agrega las features de texto (réplica del ranker del grupo)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pool-por-texto", action="store_true",
                   help="elige las --pool candidatas con B0 + texto en vez de solo B0 (el LightGBM reordena ese pool)")
    g.add_argument("--pool-por-b0", action="store_true",
                   help="elige las --pool candidatas solo con B0 (modo anterior); útil si POOL_POR_TEXTO_DEFECTO está en True")
    ap.add_argument("--modo", choices=["validar", "final"], default="validar")
    ap.add_argument("--corte", default="2024-01-01")
    ap.add_argument("--test-raiz", default=None, help="final: raíz con <vertical>/licitaciones.parquet y empresas.parquet del test")
    ap.add_argument("--salida-dir", default=".")
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=2500)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--verificar", action="store_true")
    args = ap.parse_args()
    args.pool = max(args.pool, mr.TOP)
    if args.pool_por_texto and args.sin_texto:
        ap.error("--pool-por-texto necesita las features de texto: no se puede combinar con --sin-texto")
    OPCIONES["pool_texto"] = bool(args.pool_por_texto or (POOL_POR_TEXTO_DEFECTO and not args.pool_por_b0 and not args.sin_texto))
    print(f"[modo] pool por {'B0 + texto (modo B)' if OPCIONES['pool_texto'] else 'B0 solo (modo A)'}", flush=True)

    dirs = list(args.datos or [])
    if args.todas:
        dirs += sorted(str(p.parent) for p in pathlib.Path("data").glob("*/licitaciones.parquet"))
    if not dirs:
        ap.error("indicá --datos <carpeta ...> o --todas")
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    code = 0
    for d in dict.fromkeys(dirs):
        try:
            preparar_vertical(d, cache, texto=not args.sin_texto)
            mr.procesar(d, args)
            if args.modo == "final" and args.test_raiz:      # la entrega pide rd.csv, d.csv, ... (no ranking_<V>.csv)
                nombre = pathlib.Path(d).name
                orig = pathlib.Path(args.salida_dir) / f"ranking_{nombre}.csv"
                if orig.exists():
                    shutil.copyfile(orig, pathlib.Path(args.salida_dir) / f"{nombre}.csv")
                    log(f"Archivo de entrega: {pathlib.Path(args.salida_dir) / (nombre + '.csv')}")
        except mr.PipelineError as e:
            print(f"ERROR [{d}]: {e}", file=sys.stderr)
            code = 2
        except Exception as e:
            print(f"ERROR INESPERADO [{d}] ({type(e).__name__}): {e}", file=sys.stderr)
            traceback.print_exc()
            code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
