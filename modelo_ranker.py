#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas>=2.0", "numpy>=1.24", "lightgbm>=4.0", "pyarrow>=14", "scikit-learn>=1.0"]
# ///
"""
LGBMRanker (lambdarank) sobre el B0 del repo: ¿quién gana la licitación?

Ponerlo en la raíz del repo (junto a b0.py y evaluar.py). Ejemplos:

    # validación temporal (entrena con t < corte, valida con t >= corte)
    uv run modelo_ranker.py --datos data/j --corte 2024-01-01 --verificar
    uv run modelo_ranker.py --todas

    # final: entrena con todo y rankea las licitaciones de <raiz>/<vertical>/
    uv run modelo_ranker.py --datos data/j --modo final --test-raiz test_data

Salida: ranking_<vertical>.csv  (id, parent_uei, puesto; hasta 100 por licitación).

Ideas clave
  * p_b0 usa las constantes y funciones OFICIALES de b0.py (NIVELES, PESOS, TAU_ANIOS,
    set_aside, semestre). Se calcula de forma vectorizada (cumsum por grupo + searchsorted)
    en vez de llamar a B0.p_b0 licitación por licitación; --verificar lo contrasta con el
    B0 oficial.
  * Sin fuga: cada licitación sólo ve contratos con fecha ESTRICTAMENTE < t, y las
    candidatas son las de su semestre (igual que evaluar.py). El split de validación es
    temporal y el early stopping usa las licitaciones más nuevas del propio train.
  * Desempate reproducible: semilla = SHA256(id de la licitación) mezclada con la empresa.
"""
import argparse
import hashlib
import sys
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

try:
    import b0  # baseline oficial del repo
except ImportError as e:  # pragma: no cover
    sys.exit("ERROR: no se pudo importar b0.py. Ejecutá el script desde la raíz del repo "
             f"(junto a b0.py). Detalle: {e}")

T0 = time.time()
TOP = 100                 # máx. empresas por licitación en la salida
BIG = 1 << 17             # base del compuesto (grupo * BIG + día); cubre ~358 años
MAX_PARES = 2_000_000     # tope de pares (licitación x empresa) por bloque denso
NOMBRES = ["_".join(c) if c else "global" for c in b0.NIVELES]
RECIENTES = {"psc_office", "psc", "global"}   # niveles con días-desde-último y conteo 365d
CONTEXTO = {"psc_office", "psc"}              # niveles con N de contratos previos de la licitación

ALIAS = {
    "lic": {"id": ["id", "tender_id", "licitacion_id", "licitacion"],
            "t": ["t", "fecha", "date", "fecha_publicacion", "timestamp"],
            "office": ["office", "oficina"], "psc": ["psc"], "naics": ["naics"],
            "set_aside": ["set_aside", "setaside", "set_aside_type"]},
    "hist": {"parent_uei": ["parent_uei", "uei", "empresa", "company"],
             "fecha": ["fecha", "t", "date", "fecha_adjudicacion", "award_date"],
             "office": ["office", "oficina"], "psc": ["psc"], "naics": ["naics"],
             "set_aside": ["set_aside", "setaside", "set_aside_type"]},
    "emp": {"parent_uei": ["parent_uei", "uei", "empresa", "company"],
            "semestre": ["semestre", "semester"]},
    "gan": {"id": ["id", "tender_id", "licitacion_id"],
            "parent_uei": ["parent_uei", "uei", "empresa", "company"]},
}
OPCIONALES = {"naics", "set_aside", "semestre"}


class PipelineError(Exception):
    """Error controlado con mensaje claro."""


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def warn(m):
    print(f"[{time.time() - T0:6.1f}s] AVISO: {m}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Carga + detección automática de columnas
# --------------------------------------------------------------------------- #
def leer(path):
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        raise PipelineError(f"No existe '{path}'.") from None
    except Exception as e:
        raise PipelineError(f"No se pudo leer '{path}': {e}") from e


def normalizar(df, tabla, nombre):
    """Renombra las columnas reales a los nombres canónicos (evita KeyError)."""
    lower = {str(c).lower(): c for c in df.columns}
    ren = {}
    for canon, aliases in ALIAS[tabla].items():
        real = next((lower[a] for a in aliases if a in lower), None)
        if real is None:  # coincidencia parcial (alias largos para no confundir con 't')
            real = next((o for a in aliases if len(a) > 2 for lc, o in lower.items() if a in lc), None)
        if real is None:
            if canon in OPCIONALES:
                continue
            raise PipelineError(f"{nombre}: no encuentro la columna '{canon}'. "
                                f"Columnas: {list(df.columns)}")
        ren[real] = canon
    out = df.rename(columns=ren)
    if out.columns.duplicated().any():
        raise PipelineError(f"{nombre}: columnas duplicadas tras normalizar {list(out.columns)}")
    for c in ("naics", "set_aside"):
        if c not in out.columns and tabla in ("lic", "hist"):
            warn(f"{nombre}: sin columna '{c}', se asume vacía.")
            out[c] = np.nan
    return out


def a_dias(s):
    """Serie fecha -> días enteros (misma truncación a 'D' que usa b0.py)."""
    x = pd.to_datetime(s)
    if getattr(x.dt, "tz", None) is not None:
        x = x.dt.tz_convert(None)
    return x.to_numpy(dtype="datetime64[ns]").astype("datetime64[D]").astype(np.int64)


def sem_de(t):
    """Semestre vectorizado ('2024S1'/'2024S2'): misma regla que b0.semestre."""
    x = pd.to_datetime(t)
    return (x.dt.year.astype(str) + "S" + np.where(x.dt.month <= 6, "1", "2")).to_numpy()


def mapear_sa(s):
    """b0.set_aside aplicado sólo a los valores únicos (rápido)."""
    s = s.astype(object).where(s.notna(), "")
    mp = {u: b0.set_aside(u) for u in pd.unique(s)}
    return s.map(mp).to_numpy()


def normas(df):
    """Columnas de nivel normalizadas igual que b0.B0 / b0.B0._p."""
    return pd.DataFrame({
        "psc": df.psc.astype(str).str.upper().to_numpy(),
        "office": df.office.astype(str).str.upper().to_numpy(),
        "naics": df.naics.astype(str).to_numpy(),
        "sa": mapear_sa(df.set_aside),
    })


def cargar(d):
    d = Path(d)
    lic = normalizar(leer(d / "licitaciones.parquet"), "lic", f"{d.name}/licitaciones")
    emp = normalizar(leer(d / "empresas.parquet"), "emp", f"{d.name}/empresas")
    hist = normalizar(leer(d / "historial.parquet"), "hist", f"{d.name}/historial")
    gp = d / "ganadores.parquet"
    gan = normalizar(leer(gp), "gan", f"{d.name}/ganadores") if gp.exists() else None
    lic["t"] = pd.to_datetime(lic["t"])
    if lic["t"].isna().any():
        warn(f"{int(lic['t'].isna().sum())} licitaciones sin fecha: se descartan.")
        lic = lic[lic["t"].notna()]
    hist = hist[hist["fecha"].notna() & hist["parent_uei"].notna()]
    return lic.reset_index(drop=True), emp, hist.reset_index(drop=True), gan


# --------------------------------------------------------------------------- #
# Utilidades de hash / mezcla (vectorizadas)
# --------------------------------------------------------------------------- #
def splitmix64(x):
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def semillas(ids):
    """Semilla uint64 por licitación = SHA256(str(id)) (un hash por id único)."""
    u, inv = np.unique(np.asarray(ids).astype(str), return_inverse=True)
    s = np.fromiter((int.from_bytes(hashlib.sha256(x.encode()).digest()[:8], "big") for x in u),
                    dtype=np.uint64, count=len(u))
    return s[inv]


# --------------------------------------------------------------------------- #
# Acumulados por grupo: "suma/conteo/último de contratos del grupo con t < tq"
# --------------------------------------------------------------------------- #
class GroupCum:
    def __init__(self, gid, ts_off, w):
        comp = gid * BIG + ts_off
        o = np.argsort(comp, kind="stable")
        self.comp, self.gid, self.ts = comp[o], gid[o], ts_off[o]
        ws = w[o]
        c = np.cumsum(ws, dtype=np.longdouble)
        inicio = np.flatnonzero(np.r_[True, self.gid[1:] != self.gid[:-1]])
        tam = np.diff(np.r_[inicio, len(ws)])
        st = np.repeat(inicio, tam)
        self.cg = (c - (c[st] - ws[st])).astype(np.float64)       # suma local acumulada
        self.rank = (np.arange(len(ws)) - st + 1).astype(np.float64)  # conteo local acumulado

    def query(self, gid_q, tq_off):
        valid = gid_q >= 0
        q = np.where(valid, gid_q, 0) * BIG + np.clip(tq_off, 0, BIG - 1)
        j = np.searchsorted(self.comp, q, side="left") - 1      # último con comp < q  (t estricto)
        ok = valid & (j >= 0)
        jj = np.where(ok, j, 0)
        ok &= self.gid[jj] == gid_q
        return (np.where(ok, self.cg[jj], 0.0), np.where(ok, self.rank[jj], 0.0),
                np.where(ok, self.ts[jj], -1))


class Indice:
    """Índice del historial: reproduce b0.B0 de forma vectorizada."""

    def __init__(self, hist, universo):
        self.universo, self.ncomp = universo, len(universo)
        if hist.empty:
            raise PipelineError("El historial está vacío.")
        n = normas(hist)
        ts = a_dias(hist["fecha"])
        self.ts0 = int(ts.min())
        ts_off = ts - self.ts0 + 1
        if ts_off.max() > BIG - 800:
            raise PipelineError("Rango de fechas del historial demasiado grande.")
        w = np.exp((ts - ts.max()) / (b0.TAU_ANIOS * 365.25))     # igual que b0.B0
        comp = np.searchsorted(universo, hist["parent_uei"].astype(str).to_numpy())
        self.keys, self.dens, self.nums = [], [], []
        for cols in b0.NIVELES:
            if cols:
                kd = n[list(cols)].drop_duplicates().reset_index(drop=True)
                kd["_g"] = np.arange(len(kd), dtype=np.int64)
                g = n[list(cols)].merge(kd, on=list(cols), how="left")["_g"].to_numpy(np.int64)
            else:
                kd, g = None, np.zeros(len(n), np.int64)
            self.keys.append(kd)
            self.dens.append(GroupCum(g, ts_off, w))
            self.nums.append(GroupCum(g * self.ncomp + comp, ts_off, w))

    def contexto(self, lic):
        """Pre-cálculo por licitación: grupos de cada nivel, fecha y denominadores."""
        n = normas(lic)
        tq = a_dias(lic["t"]) - self.ts0 + 1
        gids, den_s, den_n = [], [], []
        for k, cols in enumerate(b0.NIVELES):
            if cols:
                g = (n[list(cols)].merge(self.keys[k], on=list(cols), how="left")["_g"]
                     .fillna(-1).to_numpy(np.int64))
            else:
                g = np.zeros(len(n), np.int64)
            s, c, _ = self.dens[k].query(g, tq)
            gids.append(g), den_s.append(s), den_n.append(c)
        return SimpleNamespace(tq=tq, gid=gids, den_s=den_s, den_n=den_n,
                               ids=lic["id"].to_numpy(), seed=semillas(lic["id"].to_numpy()),
                               sem=sem_de(lic["t"]))

    def p_denso(self, ctx, lpos, cs):
        """p_b0 (L x C) de las licitaciones `lpos` contra las candidatas `cs`."""
        L, C = len(lpos), len(cs)
        lrep, crep = np.repeat(lpos, C), np.tile(cs, L)
        tq = ctx.tq[lrep]
        p = np.zeros(L * C)
        for k in range(len(b0.NIVELES)):
            gq = ctx.gid[k][lrep]
            s, _, _ = self.nums[k].query(np.where(gq >= 0, gq * self.ncomp + crep, -1), tq)
            den = ctx.den_s[k][lrep]
            p += b0.PESOS[k] * np.divide(s, den, out=np.zeros_like(s), where=den > 0)
        return np.round(p, 12).reshape(L, C)

    def feats_pares(self, ctx, lsel, csel):
        """Estadísticas históricas (sólo pasado) para los pares seleccionados."""
        tq, out, ptot = ctx.tq[lsel], {}, 0.0
        for k, nm in enumerate(NOMBRES):
            gq = ctx.gid[k][lsel]
            pg = np.where(gq >= 0, gq * self.ncomp + csel, -1)
            s, n, last = self.nums[k].query(pg, tq)
            den = ctx.den_s[k][lsel]
            P = np.divide(s, den, out=np.zeros_like(s), where=den > 0)
            out[f"P_{nm}"], out[f"n_{nm}"] = P, n
            ptot = ptot + b0.PESOS[k] * P
            if nm in RECIENTES:
                out[f"dias_{nm}"] = np.where(last > 0, tq - last, np.nan)
                out[f"rec365_{nm}"] = n - self.nums[k].query(pg, tq - 365)[1]
            if nm in CONTEXTO:
                out[f"dn_{nm}"] = ctx.den_n[k][lsel]
        out["p_b0"] = np.round(ptot, 12)
        return out


# --------------------------------------------------------------------------- #
# Candidatas (semestre) y construcción del pool por licitación
# --------------------------------------------------------------------------- #
def preparar_emp(emp, universo):
    e = emp.copy()
    tiene = "semestre" in e.columns
    e["_c"] = np.searchsorted(universo, e["parent_uei"].astype(str).to_numpy())
    e["_sem"] = e["semestre"].astype(str) if tiene else "*"
    e = e.drop_duplicates(["_sem", "_c"]).sort_values(["_sem", "_c"]).reset_index(drop=True)
    nums = [c for c in e.columns if c not in ("parent_uei", "semestre", "_c", "_sem")
            and "uei" not in str(c).lower() and pd.api.types.is_numeric_dtype(e[c])]
    feat = e[["_sem", "_c"] + nums].rename(columns={c: f"emp_{c}" for c in nums})
    feat[[f"emp_{c}" for c in nums]] = feat[[f"emp_{c}" for c in nums]].astype("float32")
    by_sem = {s: g["_c"].to_numpy(np.int64) for s, g in e.groupby("_sem")}
    return SimpleNamespace(tiene_sem=tiene, by_sem=by_sem, feat=feat)


def construir(idx, ctx, ei, pos, pool, universo, gan=None):
    """Pool (top-`pool` por p_b0 con desempate SHA256) + features, para las licitaciones `pos`."""
    sem = ctx.sem[pos] if ei.tiene_sem else np.full(len(pos), "*")
    bloques, sin_cands = [], 0
    for s in pd.unique(sem):                                  # <= ~14 semestres
        lp = pos[sem == s]
        cs = ei.by_sem.get(s)
        if cs is None or len(cs) == 0:
            sin_cands += len(lp)
            continue
        cu = splitmix64(cs.astype(np.uint64))
        paso = max(1, MAX_PARES // len(cs))
        K = min(pool, len(cs))
        for i in range(0, len(lp), paso):
            lb = lp[i:i + paso]
            p = idx.p_denso(ctx, lb, cs)
            tie = splitmix64(ctx.seed[lb][:, None] ^ cu[None, :])
            orden = np.lexsort((tie, -p), axis=1)[:, :K]      # p desc, luego desempate hash
            lsel, csel = np.repeat(lb, K), cs[orden].ravel()
            f = idx.feats_pares(ctx, lsel, csel)
            bloques.append(pd.DataFrame({"lic_pos": lsel, "_c": csel, "_sem": s,
                                         "rank_b0": np.tile(np.arange(1, K + 1), len(lb)), **f}))
    if sin_cands:
        warn(f"{sin_cands} licitaciones sin candidatas en su semestre (no se pueden rankear).")
    if not bloques:
        raise PipelineError("Ninguna licitación tiene candidatas.")
    df = pd.concat(bloques, ignore_index=True)
    df = df.sort_values(["lic_pos", "rank_b0"], kind="stable").reset_index(drop=True)
    df["p_rel"] = df["p_b0"] / df.groupby("lic_pos")["p_b0"].transform("max").replace(0, np.nan)
    df["p_rel"] = df["p_rel"].fillna(0.0)
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


NO_FEAT = {"lic_pos", "_c", "_sem", "id", "parent_uei", "label", "tq"}


def columnas_feat(df):
    return [c for c in df.columns if c not in NO_FEAT]


# --------------------------------------------------------------------------- #
# Modelo y ranking
# --------------------------------------------------------------------------- #
def entrenar(df, args):
    import lightgbm as lgb
    df = df[df.groupby("lic_pos")["label"].transform("max") > 0]   # grupos sin ganador no enseñan
    if df["lic_pos"].nunique() < 30:
        raise PipelineError("Muy pocas licitaciones de entrenamiento con el ganador en el pool "
                            f"({df['lic_pos'].nunique()}). Probá --pool más grande o más --max-train.")
    feats = columnas_feat(df)
    tq_g = df.groupby("lic_pos", sort=False)["tq"].first()
    thr = tq_g.quantile(0.85)                                      # 15% más nuevo -> early stopping
    es = (df["tq"] > thr).to_numpy()
    tr = ~es
    params = dict(objective="lambdarank", n_estimators=600, learning_rate=0.05, num_leaves=31,
                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  reg_lambda=1.0, random_state=args.semilla, n_jobs=-1, deterministic=True,
                  force_row_wise=True, verbose=-1)
    m = lgb.LGBMRanker(**params)

    def grp(x):
        return x.groupby("lic_pos", sort=False).size().to_numpy()

    d_tr, d_es = df[tr], df[es]
    if d_es["lic_pos"].nunique() >= 20 and d_tr["lic_pos"].nunique() >= 30:
        m.fit(d_tr[feats].astype("float32"), d_tr["label"], group=grp(d_tr),
              eval_set=[(d_es[feats].astype("float32"), d_es["label"])], eval_group=[grp(d_es)],
              eval_at=[10], callbacks=[lgb.early_stopping(40, first_metric_only=True, verbose=False)])
        log(f"Entrenado: {d_tr['lic_pos'].nunique()} licitaciones, mejor iteración={m.best_iteration_}")
    else:
        m.set_params(n_estimators=200)
        m.fit(df[feats].astype("float32"), df["label"], group=grp(df))
        log(f"Entrenado sin early stopping: {df['lic_pos'].nunique()} licitaciones")
    return m, feats


def armar_ranking(df, score, ctx, top=TOP):
    """Orden final: score desc; empates -> clave SHA256(id) mezclada con la empresa."""
    d = df[["lic_pos", "id", "parent_uei", "_c"]].copy()
    d["score"] = np.asarray(score, dtype="float64")
    d["tie"] = splitmix64(ctx.seed[d["lic_pos"].to_numpy()] ^
                          splitmix64(d["_c"].to_numpy().astype(np.uint64)))
    d = d.sort_values(["lic_pos", "score", "tie"], ascending=[True, False, True], kind="stable")
    d["puesto"] = d.groupby("lic_pos", sort=False).cumcount() + 1
    return d.loc[d["puesto"] <= top, ["id", "parent_uei", "puesto"]].reset_index(drop=True)


def verificar(idx, ctx, ei, lic, emp, hist, universo, n=5):
    """Contrasta p_b0 vectorizado con b0.B0(...).p_b0 oficial en n licitaciones."""
    prior = b0.B0(hist)
    pos = np.unique(np.linspace(0, len(lic) - 1, n).astype(int))
    peor = 0.0
    for p in pos:
        row = next(lic.iloc[[p]].itertuples())
        off = prior.p_b0(row, emp)
        off = off[~off.index.duplicated()]
        cs = ei.by_sem.get(ctx.sem[p] if ei.tiene_sem else "*")
        if cs is None:
            continue
        mio = pd.Series(idx.p_denso(ctx, np.array([p]), cs)[0], index=universo[cs])
        d = (mio.reindex(off.index).fillna(-1.0) - off).abs().max()
        peor = max(peor, float(d))
    (log if peor < 1e-6 else warn)(f"Verificación vs b0.B0 oficial: diferencia máx. de p_b0 = {peor:.2e}")


# --------------------------------------------------------------------------- #
# Orquestación por vertical
# --------------------------------------------------------------------------- #
def procesar(vdir, args):
    nombre = Path(vdir).name
    log(f"=== {nombre} ===")
    lic, emp, hist, gan = cargar(vdir)
    if gan is None:
        raise PipelineError(f"{nombre}: falta ganadores.parquet (necesario para entrenar).")

    test = None
    if args.modo == "final" and args.test_raiz:
        td = Path(args.test_raiz) / nombre
        lic_t, emp_t, hist_t, _ = cargar_test(td)
        test = (lic_t, emp_t, hist_t if hist_t is not None else hist)

    universo = np.unique(np.concatenate([
        hist["parent_uei"].astype(str).to_numpy(), emp["parent_uei"].astype(str).to_numpy(),
        *([] if test is None else [test[1]["parent_uei"].astype(str).to_numpy(),
                                  test[2]["parent_uei"].astype(str).to_numpy()])]))
    idx = Indice(hist, universo)
    ei = preparar_emp(emp, universo)
    ctx = idx.contexto(lic)
    log(f"Índice listo: {len(hist):,} contratos, {len(lic):,} licitaciones, {len(universo):,} empresas")

    if args.verificar:
        verificar(idx, ctx, ei, lic, emp, hist, universo)

    # ---- split temporal ----
    orden = np.argsort(lic["t"].to_numpy(), kind="stable")
    if args.modo == "validar":
        corte = pd.Timestamp(args.corte)
        t = lic["t"].to_numpy()
        pos_tr = orden[t[orden] < np.datetime64(corte)]
        pos_va = orden[t[orden] >= np.datetime64(corte)]
        if len(pos_va) == 0 or len(pos_tr) < 50:
            raise PipelineError(f"Con corte {args.corte}: train={len(pos_tr)}, val={len(pos_va)}. "
                                "Ajustá --corte.")
    else:
        pos_tr, pos_va = orden, None
    pos_tr = pos_tr[-args.max_train:]                          # las más recientes antes del corte
    log(f"Train: {len(pos_tr):,} licitaciones | " +
        (f"Validación: {len(pos_va):,}" if pos_va is not None else "modo final"))

    df_tr = construir(idx, ctx, ei, pos_tr, args.pool, universo, gan)
    log(f"Pool de entrenamiento: {len(df_tr):,} filas, Hit@pool={df_tr.groupby('lic_pos')['label'].max().mean():.3f}")
    modelo, feats = entrenar(df_tr, args)

    # ---- predicción ----
    if pos_va is not None:
        df_te = construir(idx, ctx, ei, pos_va, args.pool, universo, gan)
        ctx_te, lic_eval, emp_eval = ctx, lic, emp
    elif test is not None:
        lic_t, emp_t, hist_t = test
        idx_t = Indice(hist_t, universo)
        ei_t = preparar_emp(emp_t, universo)
        ctx_te = idx_t.contexto(lic_t)
        df_te = construir(idx_t, ctx_te, ei_t, np.arange(len(lic_t)), args.pool, universo)
        lic_eval, emp_eval = lic_t, emp_t
    else:
        warn("Modo final sin --test-raiz: se rankean las licitaciones de entrenamiento (sólo sanity check).")
        df_te = construir(idx, ctx, ei, np.arange(len(lic)), args.pool, universo, gan)
        ctx_te, lic_eval, emp_eval = ctx, lic, emp

    for c in feats:                                            # columnas ausentes (p. ej. emp_*)
        if c not in df_te.columns:
            df_te[c] = np.nan
    score = modelo.predict(df_te[feats].astype("float32"))
    rk = armar_ranking(df_te, score, ctx_te)
    out = Path(args.salida_dir)
    out.mkdir(parents=True, exist_ok=True)
    f_out = out / f"ranking_{nombre}.csv"
    rk.to_csv(f_out, index=False)
    log(f"Ranking guardado: {f_out} ({rk['id'].nunique():,} licitaciones, {len(rk):,} filas)")

    # ---- evaluación (sólo validación) con evaluar.py del repo ----
    if pos_va is not None:
        ids_va = set(lic["id"].iloc[pos_va])
        gan_va = gan[gan["id"].isin(ids_va)][["id", "parent_uei"]]
        gan_va.to_parquet(out / f"mi_val_{nombre}.parquet")
        rk_b0 = armar_ranking(df_te, -df_te["rank_b0"].to_numpy(), ctx_te)
        rk_b0.to_csv(out / f"b0_val_{nombre}.csv", index=False)
        try:
            import evaluar as ev
            res = ev.evaluar(rk[rk["id"].isin(ids_va)], lic_eval, emp_eval, gan_va, rk_b0)
            with pd.option_context("display.width", 220, "display.max_columns", 50,
                                   "display.float_format", "{:.3f}".format):
                print(res.to_string(index=False))
        except ImportError:
            warn("No se pudo importar evaluar.py: se omite la evaluación.")
        except Exception as e:
            warn(f"evaluar.py rechazó/falló ({type(e).__name__}): {e}")
        log("Para comparar con el B0 oficial: uv run b0.py <datos> --salida b0.csv y luego "
            f"uv run evaluar.py {f_out} --datos {vdir} --respuestas {out / f'mi_val_{nombre}.parquet'} --b0 b0.csv")


def cargar_test(td):
    td = Path(td)
    lic = normalizar(leer(td / "licitaciones.parquet"), "lic", f"{td.name}/licitaciones(test)")
    emp = normalizar(leer(td / "empresas.parquet"), "emp", f"{td.name}/empresas(test)")
    lic["t"] = pd.to_datetime(lic["t"])
    hp = td / "historial.parquet"
    hist = None
    if hp.exists():
        hist = normalizar(leer(hp), "hist", f"{td.name}/historial(test)")
        hist = hist[hist["fecha"].notna() & hist["parent_uei"].notna()].reset_index(drop=True)
    return lic.reset_index(drop=True), emp, hist, None


def main():
    ap = argparse.ArgumentParser(description="LGBMRanker sobre el B0 (licitaciones)")
    ap.add_argument("--datos", nargs="+", help="carpetas de vertical, p. ej. data/j data/rd")
    ap.add_argument("--todas", action="store_true", help="todas las carpetas data/*/ con licitaciones")
    ap.add_argument("--modo", choices=["validar", "final"], default="validar")
    ap.add_argument("--corte", default="2024-01-01", help="validar: val = licitaciones con t >= corte")
    ap.add_argument("--test-raiz", help="final: raíz con <vertical>/licitaciones.parquet + empresas.parquet")
    ap.add_argument("--salida-dir", default=".", help="carpeta de salida de los CSV")
    ap.add_argument("--pool", type=int, default=200, help="candidatas por licitación que ve el modelo (>=100)")
    ap.add_argument("--max-train", type=int, default=2500, help="máx. licitaciones de entrenamiento")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--verificar", action="store_true", help="contrasta p_b0 con b0.B0 oficial")
    args = ap.parse_args()
    args.pool = max(args.pool, TOP)

    dirs = list(args.datos or [])
    if args.todas:
        dirs += sorted(str(p.parent) for p in Path("data").glob("*/licitaciones.parquet"))
    if not dirs:
        ap.error("indicá --datos <carpeta ...> o --todas")
    code = 0
    for d in dict.fromkeys(dirs):
        try:
            procesar(d, args)
        except PipelineError as e:
            print(f"ERROR [{d}]: {e}", file=sys.stderr)
            code = 2
        except KeyboardInterrupt:
            print("Interrumpido.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"ERROR INESPERADO [{d}] ({type(e).__name__}): {e}", file=sys.stderr)
            traceback.print_exc()
            code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
