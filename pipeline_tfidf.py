"""
Pipeline 1: B0 + similitud de texto (TF-IDF).

Para cada licitación de validación, a cada empresa candidata se le calculan dos puntajes:
  - p   : el puntaje del B0 (quién ganó antes en esa oficina/rubro, sin leer texto)
  - sim : similitud coseno entre el texto de la licitación y el perfil de la empresa (texto_awards)
Se mezclan como  norm(p) + alpha * norm(sim)  y se queda con las `top` mejores.

Uso (desde la carpeta del repo, con data/ descargado):
    python pipeline_tfidf.py --datos data/rd
    python pipeline_tfidf.py --datos data/j --alphas 0 0.25 0.5 1
    python pipeline_tfidf.py --datos data/rd --con-b0     # agrega la comparación contra el B0

Salida: un ranking por alpha (rank_<vertical>_alpha<valor>.csv) y las métricas impresas.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, ".")   # b0.py vive en la raíz del repo
import b0                 # noqa: E402  (baseline de la cátedra)


# ---------- utilidades ----------
def semestre_de(t):
    """Fecha -> '2024S1' / '2024S2'."""
    t = pd.Timestamp(t)
    return f"{t.year}S{1 if t.month <= 6 else 2}"


def txt(x):
    """Texto seguro: los valores faltantes pasan a ''."""
    return "" if pd.isna(x) else str(x)


def norm(x):
    """Escala a [0, 1] dentro de una licitación, para poder sumar p y sim."""
    m = x.max()
    return x / m if m > 0 else x


# ---------- pasos del pipeline ----------
def cargar(datos):
    d = Path(datos)
    return (pd.read_parquet(d / "licitaciones.parquet"),
            pd.read_parquet(d / "ganadores.parquet"),
            pd.read_parquet(d / "empresas.parquet"),
            pd.read_parquet(d / "historial.parquet"))


def ajustar_tfidf(hist, corte):
    """Aprende el vocabulario SOLO con textos anteriores al corte (sin mirar el futuro)."""
    pasado = hist.loc[pd.to_datetime(hist["fecha"]) < pd.Timestamp(corte), "descripcion"].fillna("")
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                            min_df=2, max_features=100_000, sublinear_tf=True)
    tfidf.fit(pasado)
    return tfidf


def puntuar(val, emp, hist, tfidf):
    """Para cada licitación: (ids de empresas, p del B0, sim de texto)."""
    prior = b0.B0(hist.copy())

    perfiles = {}   # semestre -> (candidatas, matriz TF-IDF de sus perfiles)
    for s in sorted(val["t"].map(semestre_de).unique()):
        cand = emp[emp["semestre"] == s].reset_index(drop=True)
        perfiles[s] = (cand, tfidf.transform(cand["texto_awards"].fillna("")))

    scores = {}
    for _, q in val.iterrows():
        cand, M = perfiles[semestre_de(q["t"])]
        p = prior.p_b0(q, cand).reindex(cand["parent_uei"]).fillna(0).to_numpy()
        v = tfidf.transform([txt(q["title"]) + " " + txt(q["description"])])
        sim = (M @ v.T).toarray().ravel()
        scores[q["id"]] = (cand["parent_uei"].to_numpy(), p, sim)
    return scores


def armar_ranking(scores, alpha, top=100, seed=0):
    """Mezcla p y sim con el peso alpha y devuelve el top por licitación."""
    rng = np.random.default_rng(seed)
    filas = []
    for lid, (ids, p, sim) in scores.items():
        s = norm(p) + alpha * norm(sim) + 1e-9 * rng.random(len(p))   # el ruido solo desempata
        for puesto, i in enumerate(np.argsort(-s)[:top], 1):
            filas.append((lid, ids[i], puesto))
    return pd.DataFrame(filas, columns=["id", "parent_uei", "puesto"])


def correr(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR al correr:", " ".join(cmd))
        print(r.stderr or r.stdout)
        sys.exit(1)
    return r.stdout


# ---------- programa principal ----------
def main():
    ap = argparse.ArgumentParser(description="Pipeline 1: B0 + TF-IDF")
    ap.add_argument("--datos", default="data/rd", help="carpeta de una vertical, ej. data/rd")
    ap.add_argument("--corte", default="2024-01-01", help="desde esta fecha es validación")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1, 2])
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--con-b0", action="store_true", help="comparar también contra el B0")
    a = ap.parse_args()

    vertical = Path(a.datos).name
    lic, gan, emp, hist = cargar(a.datos)

    val = lic[lic["t"] >= a.corte].copy()
    if val.empty:
        sys.exit(f"No hay licitaciones desde {a.corte}.")
    print(f"vertical={vertical} | licitaciones de validación: {len(val)}")

    tfidf = ajustar_tfidf(hist, a.corte)
    print("vocabulario TF-IDF:", len(tfidf.vocabulary_))
    scores = puntuar(val, emp, hist, tfidf)
    print("licitaciones puntuadas:", len(scores))

    with tempfile.TemporaryDirectory() as tmp:       # la "hoja de respuestas" es temporal
        respuestas = str(Path(tmp) / "val.parquet")
        gan[gan["id"].isin(val["id"])].to_parquet(respuestas)

        extra = []
        if a.con_b0:
            b0_csv = f"b0_{vertical}.csv"
            correr([sys.executable, "b0.py", a.datos, "--salida", b0_csv])
            extra = ["--b0", b0_csv]

        for alpha in a.alphas:
            archivo = f"rank_{vertical}_alpha{alpha}.csv"
            armar_ranking(scores, alpha, a.top).to_csv(archivo, index=False)
            salida = correr([sys.executable, "evaluar.py", archivo, "--datos", a.datos,
                             "--respuestas", respuestas] + extra)
            print(f"--- alpha = {alpha} ---")
            print(salida)


if __name__ == "__main__":
    main()
