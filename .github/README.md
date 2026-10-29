# Grupo F01: predicción de la empresa ganadora de una licitación

Integrantes: Luis Fernando Morales Rivera, Abigail Freijomil y Rebecca Lis Salvati.

Para cada licitación, el modelo ordena hasta 100 empresas candidatas de la más a la menos probable de ganar. Se evalúa con el MRR
(promedio del inverso del puesto de la ganadora) en cinco verticales: `rd`, `d`, `it`, `r` y `j`. El punto de partida es el baseline B0
de la cátedra (`b0.py`), que puntúa a cada empresa por su historial de contratos.

## Qué hace nuestro método

1. **Ranker con aprendizaje.** Un LightGBM de ranking (lambdarank), entrenado por vertical, reordena un pool de 200 candidatas por licitación.
   Usa el puntaje del B0 y rasgos del historial de cada empresa (`modelo_ranker.py`, desarrollado por Luis Morales Rivera).
2. **Texto.** Le sumamos como rasgos la similitud entre el texto de la licitación y el de los contratos anteriores de cada empresa, con TF-IDF y
   con embeddings de oraciones (all-MiniLM-L6-v2), absolutas y relativas al resto del pool (`modelo_ranker_texto.py`).
3. **Pool.** El único cambio entre los dos modos es cómo se elige el pool de 200 que reordena el ranker:

| modo | pool de 200 | cómo se pide |
|---|---|---|
| **B** (por defecto, el de la entrega) | las de mejor B0 + similitud de texto | sin banderas, o `--pool-por-texto` |
| A | las de mejor B0 (solo historial) | `--pool-por-b0` |

El defecto se cambia con la constante `POOL_POR_TEXTO_DEFECTO` al comienzo de `modelo_ranker_texto.py`.

Sin fuga de información: de cada empresa solo se mira lo anterior a la licitación, y el vocabulario del texto y los perfiles de embeddings
se arman solo con contratos anteriores al período que se predice.

## Resultados de validación

Se entrena con las licitaciones anteriores al 1/1/2024 y se valida en 2024S1: 2000 licitaciones de las cinco verticales juntas.

| | B0 | modo A | modo B | B menos A (IC95 por bootstrap pareado) |
|---|--:|--:|--:|---|
| MRR | 0,139 | 0,258 | 0,273 | +0,014 (+0,005 a +0,024) |

El detalle por vertical y los experimentos anteriores (TF-IDF, embeddings, SVD, lematizado, top protegido) están en `resultados.md` y en el informe.

## Reproducir las predicciones del test

Requiere Python 3.10+ y, para codificar las licitaciones del test con MiniLM, de preferencia una GPU (en una T4 de Colab la corrida de las cinco
verticales tarda unos 25 minutos). Desde la raíz del repo:

```
pip install -r requirements.txt     # en Colab alcanza con: pip install lightgbm==4.6.0 sentence-transformers==5.7.0 pyarrow

# 1. datos y test de la cátedra
curl -LO https://github.com/bravoivanagustin/ranking-licitaciones/releases/download/datos-v1/data.zip && unzip -oq data.zip
curl -LO https://github.com/bravoivanagustin/ranking-licitaciones/releases/download/test-v1/test.zip && unzip -oq test.zip

# 2. vectores MiniLM de los contratos del historial (release vectores-v1 de este repo); si no se bajan, se calculan al correr (más de 1 hora)
mkdir -p cache
for V in rd d it r j; do curl -sfL https://github.com/AbiFrei/grupo_Luabre/releases/download/vectores-v1/emb_hist_$V.npy -o cache/emb_hist_$V.npy; done

# 3. predicciones: escribe entrega/rd.csv, d.csv, it.csv, r.csv y j.csv (id, parent_uei, puesto; hasta 100 por licitación)
python modelo_ranker_texto.py --todas --cache cache --modo final --test-raiz test --salida-dir entrega

# 4. verificación del formato contra el test
python verificar_entrega.py --csv-dir entrega --test-raiz test
```

`python EJECUTABLE.py` hace estos cuatro pasos de una sola vez (instala, baja datos, test y vectores comprobando que no queden cortados, corre el modelo y verifica). Los comandos de arriba son el camino principal para reproducir: funcionan en cualquier computadora con Python. El notebook `ejecutable_NB_colab.ipynb` ([abrirlo en Colab](https://colab.research.google.com/github/AbiFrei/grupo_Luabre/blob/main/ejecutable_NB_colab.ipynb)) es solo una comodidad: hace lo mismo en Colab, de los datos a los cinco CSV, y además valida el método. Se ofrece porque la codificación del texto con MiniLM necesita una GPU para tardar minutos y no horas, y Colab la presta gratis (T4); no hace nada que los comandos no hagan.
Con `--pool-por-b0` y otra carpeta de salida se obtiene el modo A.

## Validación (sin el test)

`python modelo_ranker_texto.py --todas --cache cache --salida-dir rankings_validacion` entrena con lo anterior a 2024-01-01 y valida en 2024S1;
con `--corte 2023-07-01` se mueve la ventana. `python comparar_modos.py --a <carpeta modo A> --b <carpeta modo B>` compara los dos modos
(MRR y diferencia con IC95). `python simular_test.py` arma un test simulado con el mismo formato que el oficial para ensayar el flujo.

## Archivos

| archivo | qué es |
|---|---|
| `modelo_ranker_texto.py` | el modelo final: ranker + features de texto, modos A y B, validación y entrega |
| `modelo_ranker.py` | ranker del historial desarrollado por Luis Morales Rivera; `modelo_ranker_texto.py` lo usa como base |
| `experimento.py` | TF-IDF determinista (`TfidfEstable`) y la etapa de B0 + texto |
| `verificar_entrega.py` | comprueba los cinco CSV contra el test antes de entregar |
| `EJECUTABLE.py` | corre toda la entrega de punta a punta con un solo comando, sin Colab |
| `comparar_modos.py`, `comparar_ranker.py` | comparaciones con IC95 entre modos y contra el ranker sin texto |
| `simular_test.py` | arma un test simulado con el formato del test oficial |
| `prueba_pool.py` | cuántas ganadoras entran al pool de 200 según cómo se elige |
| `experimento_emb.py`, `experimento_top.py`, `analisis_alpha.py`, `pipeline_tfidf.py` | experimentos de la primera etapa (B0 + texto) |
| `b0.py`, `evaluar.py` | baseline y evaluador de la cátedra |
| `resultados.md` | registro de todos los experimentos |
| `ejecutable_NB_colab.ipynb` | notebook de Colab que reproduce todo |

## Reproducibilidad

Los desempates y las semillas son fijos (semilla 42; desempate por SHA-256 del id). En el test se usa el historial de `data/` (termina el
30/6/2024) y la tabla de empresas del test, que no trae `semestre`. Dos corridas independientes dieron el mismo pool de entrenamiento y la misma
iteración final del LightGBM en cada vertical.

El `README.md` de la raíz del repo es el enunciado de la cátedra (datos y formato).
