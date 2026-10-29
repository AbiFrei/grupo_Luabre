# Grupo F01: solución y cómo reproducir las predicciones

Método: **ranker de historial (LightGBM, lambdarank) + features de texto** (similitud TF-IDF y embeddings MiniLM),
sobre un pool combinado de 200 candidatas por licitación, elegido con B0 + similitud de texto.
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

`python EJECUTABLE.py` hace estos cuatro pasos de una sola vez (instala, baja datos, test y vectores comprobando que no queden cortados, corre el modelo y verifica). Los comandos de arriba son el camino principal para reproducir: funcionan en cualquier computadora con Python. El notebook `ejecutable_NB_colab.ipynb` ([abrirlo en Colab](https://colab.research.google.com/github/AbiFrei/grupo_Luabre/blob/main/ejecutable_NB_colab.ipynb)) es solo una comodidad: hace lo mismo en Colab, de los datos a los cinco CSV, y además valida el método. Se ofrece porque la codificación del texto con MiniLM necesita una GPU para tardar minutos y no horas, y Colab la presta gratis (T4); no hace nada que los comandos no hagan.

## El pool combinado

Un ranker solo puede reordenar a quien entró al pool. El pool de 200 candidatas por licitación no se arma con el B0 solo (historial), sino con la
combinación de B0 y similitud de texto (TF-IDF y embeddings), de modo que entran más ganadoras al pool que reordena el LightGBM.

## Validación (sin el test)

`python modelo_ranker_texto.py --todas --cache cache --salida-dir rankings_validacion` entrena con lo anterior a 2024-01-01 y
valida en 2024S1; con `--corte 2023-07-01` se mueve la ventana. Resultados de la validación 2024S1, cinco verticales juntas (2000 licitaciones):
B0 0,139; ranker del historial sin texto 0,172; método final (ranker + texto, pool combinado) 0,273 (+0,101 sobre el ranker sin texto, IC95 por
bootstrap pareado +0,088 a +0,115). `comparar_ranker.py` hace la comparación con el ranker sin texto. El detalle por vertical y los
experimentos anteriores están en el informe y en `resultados.md`.

## Archivos

| archivo | qué es |
|---|---|
| `modelo_ranker_texto.py` | el modelo final: ranker + features de texto con pool combinado, validación y entrega |
| `modelo_ranker.py` | ranker del historial desarrollado por Luis Morales Rivera; `modelo_ranker_texto.py` lo usa como base |
| `experimento.py` | TF-IDF determinista (`TfidfEstable`) y la etapa de B0 + texto |
| `verificar_entrega.py` | comprueba los cinco CSV contra el test antes de entregar |
| `EJECUTABLE.py` | corre toda la entrega de punta a punta con un solo comando, sin Colab |
| `comparar_ranker.py` | compara el método con el ranker sin texto y con B0 (MRR y diferencia con IC95) |
| `simular_test.py` | arma un test simulado con el mismo formato que el oficial, para ensayar el flujo |
| `prueba_pool.py` | prueba del pool sin LightGBM: cuántas ganadoras entran al pool de 200 |
| `b0.py`, `evaluar.py` | baseline y evaluador de la cátedra |

## Reproducibilidad

Los desempates y las semillas son fijos (semilla 42; desempate por SHA-256 del id). El vocabulario TF-IDF y los perfiles de
embeddings de cada empresa se calculan solo con contratos anteriores al período que se predice. En el test se usa el historial
de `data/` (termina el 30/6/2024) y la tabla de empresas del test, sin `semestre`. Dos corridas independientes dieron el mismo
pool de entrenamiento y la misma iteración final del LightGBM en cada vertical.

## Visibilidad del repositorio

Dejamos el repositorio público a propósito. El ejecutable (`EJECUTABLE.py` y el notebook de Colab) baja el código, los vectores y los datos desde GitHub y desde
los releases de este repo, y GitHub no permite esas descargas sin iniciar sesión si el repositorio es privado. Público, cualquiera puede reproducir la entrega con un
solo comando, sin credenciales ni permisos. Además, `bravoivanagustin` y `JHK212` están invitados como colaboradores.
