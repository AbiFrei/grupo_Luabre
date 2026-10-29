# Grupo F01: solución y cómo reproducir las predicciones

Método: **ranker de historial (LightGBM, lambdarank) + features de texto** (similitud TF-IDF y embeddings MiniLM),
sobre un pool de 200 candidatas por licitación elegido con B0 + texto (modo B, el de la entrega).
Integrantes: Luis Fernando Morales Rivera, Abigail Freijomil y Rebecca Lis Salvati.

## Reproducir las predicciones del test (lo mínimo)

Requiere Python 3.10+ y, para codificar las licitaciones del test con MiniLM, de preferencia una GPU (en una T4 de Colab,
la corrida de las cinco verticales tarda unos 25 minutos). Desde la raíz del repo:

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

Los comandos de arriba son el camino principal para reproducir: funcionan en cualquier computadora con Python. El notebook `ejecutable_limpio.ipynb` es solo una comodidad: hace lo mismo en Colab, de los datos a los cinco CSV, y además valida el método. Se ofrece porque la codificación del texto con MiniLM necesita una GPU para tardar minutos y no horas, y Colab la presta gratis (T4); no hace nada que los comandos no hagan.

## Los dos modos

El único cambio entre ellos es cómo se elige el pool de 200 candidatas que reordena el LightGBM:

| modo | pool de 200 | cómo se pide |
|---|---|---|
| **B** (por defecto) | las de mejor B0 + similitud de texto | sin banderas, o `--pool-por-texto` |
| A | las de mejor B0 (solo historial) | `--pool-por-b0` |

El defecto se cambia con la constante `POOL_POR_TEXTO_DEFECTO` al comienzo de `modelo_ranker_texto.py`.
Los resultados del modo A salen de agregar `--pool-por-b0` y otra carpeta de salida.

## Validación (sin el test)

`python modelo_ranker_texto.py --todas --cache cache --salida-dir rankings_validacion` entrena con lo anterior a 2024-01-01 y
valida en 2024S1; con `--corte 2023-07-01` se mueve la ventana. `python comparar_modos.py --a <carpeta modo A> --b <carpeta modo B>`
compara los dos modos (MRR y diferencia con IC95 por bootstrap pareado). Resultados de la validación 2024S1, cinco verticales juntas
(2000 licitaciones): B0 0,139, modo A 0,258, modo B 0,273 (B menos A: +0,014, IC95 +0,005 a +0,024). El detalle de los experimentos
anteriores está en `resultados.md`.

## Archivos

| archivo | qué es |
|---|---|
| `modelo_ranker_texto.py` | el modelo final: ranker + features de texto, modos A y B, validación y entrega |
| `modelo_ranker.py` | ranker del historial desarrollado por Luis Morales Rivera; `modelo_ranker_texto.py` lo usa como base |
| `experimento.py` | TF-IDF determinista (`TfidfEstable`) y la etapa de B0 + texto |
| `verificar_entrega.py` | comprueba los cinco CSV contra el test antes de entregar |
| `comparar_modos.py`, `comparar_ranker.py` | comparaciones con IC95 entre modos y contra el ranker sin texto |
| `simular_test.py` | arma un test simulado con el mismo formato que el oficial, para ensayar el flujo |
| `prueba_pool.py` | prueba del pool sin LightGBM: cuántas ganadoras entran al pool de 200 |
| `b0.py`, `evaluar.py` | baseline y evaluador de la cátedra |

## Reproducibilidad

Los desempates y las semillas son fijos (semilla 42; desempate por SHA-256 del id). El vocabulario TF-IDF y los perfiles de
embeddings de cada empresa se calculan solo con contratos anteriores al período que se predice. En el test se usa el historial
de `data/` (termina el 30/6/2024) y la tabla de empresas del test, sin `semestre`. Dos corridas independientes dieron el mismo
pool de entrenamiento y la misma iteración final del LightGBM en cada vertical.
