# Resultados de los experimentos

Registro de cada experimento: qué se probó, con qué configuración y qué métricas dio.
Se agrega una sección por experimento. Es la base del informe de enfoques estudiados.

## Experimento 1: B0 + TF-IDF (vertical `rd`)

**Configuración**
- Vertical: `rd` (328 licitaciones en total).
- Validación: licitaciones desde 2024-01-01 (**57 licitaciones**, 87 ganadores). Todo lo anterior es "pasado".
- Perfil de empresa: `texto_awards` (descripciones de sus contratos anteriores al semestre).
- Texto de la licitación: `title` + `description`.
- TF-IDF: `stop_words="english"`, `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True`; vocabulario aprendido solo con contratos anteriores al 2024-01-01.
- Puntaje final: `norm(p_b0) + alpha * norm(similitud_coseno)`, top 100.
- Código: `python pipeline_tfidf.py --datos data/rd`

**Referencia: B0 sobre todo `rd` (coincide con el README)**

| Conjunto | N | MRR | Hit@5 | Hit@1 | Hit@100 | Recall@100 | Puesto mediano |
|---|---|---|---|---|---|---|---|
| todas | 328 | 0,125 | 0,271 | 0,079 | 0,628 | 0,532 | 9 |

**Resultados en la validación (57 licitaciones)**

| alpha | MRR | Hit@5 | Hit@1 | Hit@100 | Recall@100 | Puesto mediano |
|---|---|---|---|---|---|---|
| 0 (B0 puro) | 0,091 | 0,228 | 0,053 | 0,491 | 0,409 | 7,0 |
| 0,1 | 0,106 | 0,228 | 0,070 | 0,526 | 0,462 | 7,5 |
| 0,25 | 0,110 | 0,246 | 0,070 | 0,544 | 0,470 | 8,0 |
| **0,5** | **0,114** | 0,246 | 0,070 | 0,526 | 0,456 | 8,0 |
| 1 | 0,112 | 0,228 | 0,070 | 0,509 | 0,443 | 6,0 |
| 2 | 0,097 | 0,158 | 0,053 | 0,474 | 0,398 | 10,0 |

**Lectura**
- Con `alpha = 0` se reproducen exactamente los números del B0 en la validación (MRR 0,091), lo que confirma que la evaluación está bien armada.
- Agregar texto mejora el MRR para `alpha` entre 0,1 y 1 (de 0,091 a 0,114 como máximo, unos +0,02). Con `alpha = 2` el texto domina y empeora.
- Los 87 ganadores de validación están todos entre las empresas candidatas, así que el techo de desempeño es 1,0.

**Limitaciones (importante para el informe)**
- Con 57 licitaciones cada una pesa ~1,75 puntos: el Hit@1 pasó de 3 a 4 licitaciones. La mejora se ve en los números pero es chica frente al ruido.
- `alpha` se eligió mirando esta misma validación, lo que infla un poco el resultado. Pendiente: elegirlo con un período anterior (p. ej. 2023) y reportar en 2024.
- Pendiente: repetir en una vertical grande (`j`, `r`, `d` o `it`) y reportar la mejora sobre el B0 con su intervalo (`evaluar.py ... --b0`).

## Próximos experimentos (ideas)
- [ ] Comparar contra cada contrato por separado y quedarse con el más parecido.
- [ ] Sumar el texto de `psc`, `naics` y `office` (tablas de códigos).
- [ ] Pesos distintos para título y descripción.
- [ ] Embeddings de oraciones.
- [ ] LightGBM (lambdarank) con `p_b0`, similitudes y otras señales como features.
