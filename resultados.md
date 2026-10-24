# Resultados de los experimentos

Registro de cada experimento: qué se probó, con qué configuración y qué métricas dio.
Se agrega una sección por experimento. Es la base del informe de enfoques estudiados.

## Experimentación 0 (notebook: celdas #7 a #13): B0 + TF-IDF (vertical `rd`)

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
- Los 87 ganadores de validación están todos entre las candidatas. Es por construcción (se sacó del dataset a las ganadoras que no eran candidatas), no un hallazgo.

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

## Experimentación 0, continuación: mejora sobre el B0 con intervalo (vertical `rd`)

Misma configuración que la Experimentación 0. La evaluación usa `evaluar.py` como módulo y compara
contra el ranking del B0 en las mismas 57 licitaciones de validación. El intervalo es un IC95
por bootstrap pareado (remuestreando licitaciones).

| alpha | MRR | Hit@5 | Hit@1 | Δ MRR vs B0 | IC95 del Δ | Hit@100 | Recall@100 |
|---|---|---|---|---|---|---|---|
| 0 (B0 puro) | 0,091 | 0,228 | 0,053 | 0,000 | (0,000; 0,000) | 0,491 | 0,409 |
| 0,1 | 0,106 | 0,228 | 0,070 | +0,015 | (+0,002; +0,040) | 0,526 | 0,462 |
| 0,25 | 0,110 | 0,246 | 0,070 | +0,019 | (+0,004; +0,046) | 0,544 | 0,470 |
| **0,5** | **0,114** | 0,246 | 0,070 | **+0,024** | (+0,004; +0,052) | 0,526 | 0,456 |
| 1 | 0,112 | 0,228 | 0,070 | +0,021 | (−0,008; +0,056) | 0,509 | 0,443 |
| 2 | 0,097 | 0,158 | 0,053 | +0,006 | (−0,060; +0,076) | 0,474 | 0,398 |

**Lectura**
- Con alpha entre 0,1 y 0,5 la mejora es positiva y su intervalo no incluye el 0; con 1 y 2 sí lo incluye.
- El piso del intervalo es muy chico (+0,002 a +0,004): la mejora es real pero puede ser casi nula.
- El texto aporta sobre todo abajo en la lista: Hit@1 sube en una sola licitación (3 → 4),
  mientras que Hit@100 pasa de 28 a 31 licitaciones.
- El puesto mediano no se compara entre filas: se calcula sobre conjuntos distintos de licitaciones.
- alpha = 0,5 se eligió mirando esta misma validación, así que el +0,024 es algo optimista.
  Próximo paso: elegir alpha con 2023 y medir una sola vez en 2024S1.

## Experimentación i (notebook: celdas #14 a #18): validación cronológica (selección 2023, prueba 2024S1) — vertical `rd`

**Esquema**
- Selección: licitaciones de 2023 (130 licitaciones, 217 ganadoras). Se usa para elegir alpha.
- Prueba: 2024S1 (57 licitaciones, 87 ganadoras). Se mide una sola vez con el alpha elegido.
- El vocabulario TF-IDF de cada semestre se aprende solo con contratos anteriores al inicio de ese semestre.
- Regla de elección fijada de antemano: el alpha con mayor MRR en selección.

**Selección (2023)**

| alpha | MRR | Hit@5 | Hit@1 | Δ MRR vs B0 | IC95 del Δ | Hit@100 | Recall@100 |
|---|---|---|---|---|---|---|---|
| 0 (B0) | 0,125 | 0,246 | 0,069 | 0,000 | (0,000; 0,000) | 0,638 | 0,535 |
| 0,05 | 0,124 | 0,246 | 0,069 | −0,001 | (−0,004; +0,002) | 0,631 | 0,525 |
| 0,1 | 0,126 | 0,254 | 0,069 | +0,001 | (−0,004; +0,005) | 0,615 | 0,519 |
| 0,25 | 0,127 | 0,269 | 0,069 | +0,002 | (−0,006; +0,010) | 0,585 | 0,497 |
| 0,5 | 0,131 | 0,285 | 0,069 | +0,006 | (−0,008; +0,020) | 0,577 | 0,485 |
| 0,75 | 0,134 | 0,300 | 0,077 | +0,009 | (−0,011; +0,031) | 0,577 | 0,478 |
| **1** | **0,143** | 0,269 | 0,092 | **+0,018** | (−0,010; +0,048) | 0,569 | 0,466 |
| 1,5 | 0,136 | 0,223 | 0,085 | +0,011 | (−0,027; +0,054) | 0,538 | 0,441 |
| 2 | 0,137 | 0,185 | 0,092 | +0,012 | (−0,029; +0,058) | 0,523 | 0,431 |

**Prueba (2024S1), alpha = 1**

| alpha | MRR | Hit@5 | Hit@1 | Δ MRR vs B0 | IC95 del Δ | Hit@100 | Recall@100 |
|---|---|---|---|---|---|---|---|
| 1 | 0,112 | 0,228 | 0,070 | +0,021 | (−0,008; +0,056) | 0,509 | 0,443 |

(B0 en prueba: MRR 0,091, Hit@5 0,228, Hit@1 0,053.)

**Lectura**
- Con validación limpia, la mejora de B0 + TF-IDF sobre el B0 es positiva (+0,021) pero no significativa: el IC95 incluye el 0.
- En la Experimentación 0 parecía significativa porque alpha se elegía y se medía en las mismas licitaciones.
- La señal de texto es inestable: en 2023 ayuda menos que en 2024 y, al subir alpha, baja el Hit@100
  (0,638 → 0,523): el texto saca de la lista a ganadoras que el B0 encontraba.
- Próximos pasos: variantes de la representación del texto (lematizar, SVD, embeddings, comparar contrato
  por contrato) con este mismo esquema, y repetir en una vertical grande para achicar los intervalos.

## Experimentación ii (notebook: celdas #19 a #23): lematizar y recortar a la raíz — vertical `rd`

**Qué se probó.** TF-IDF compara palabras exactas ("service" y "services" son términos distintos).
Se probaron dos formas de normalizar las palabras al tokenizar, antes de armar los vectores:
- **Lematizar** (WordNet): forma de diccionario ("laboratories" → "laboratory").
- **Recortar a la raíz** (Snowball): corta terminaciones con reglas fijas ("laboratories" → "laboratori").

Todo lo demás igual que en la Experimentación i (B0, mezcla con alpha, selección 2023, prueba 2024S1).

**Regla de decisión (fijada antes de mirar).** Cada variante usa su mejor alpha en selección y se compara contra
el TF-IDF base licitación por licitación (IC95 por bootstrap pareado). Se adopta solo si el IC95 del Δ es todo positivo.

**Resultado en selección (2023, 130 licitaciones)**

| Variante | Mejor alpha | MRR | Δ vs TF-IDF base | IC95 del Δ | Mejoran / empeoran / iguales |
|---|---|---|---|---|---|
| TF-IDF base | 1 | 0,143 | — | — | — |
| Lematizado | 1 | 0,144 | +0,001 | (−0,002; +0,006) | 27 / 21 / 82 |
| Raíz | 1 | 0,140 | −0,002 | (−0,014; +0,007) | 35 / 23 / 72 |

**Lectura**
- Lematizar da prácticamente lo mismo que el base: mejoras y empeoramientos se cancelan.
- Recortar a la raíz mejora más licitaciones de las que empeora (35 contra 23), pero el MRR promedio baja:
  mejora un poco a ganadoras que estaban abajo y empeora a unas pocas que estaban arriba
  (en el MRR, bajar del puesto 1 al 2 cuesta 0,5; subir del 50 al 20 suma 0,03).
- Ningún IC95 es todo positivo: se mantiene el TF-IDF base y no se mide nada nuevo en prueba.
  El resultado reportado sigue siendo el de la Experimentación i (MRR 0,112 contra 0,091 del B0 en 2024S1).
- Conclusión: las diferencias de forma de las palabras no son el problema principal del texto.
  La hipótesis siguiente es un problema de significado (sinónimos, temas): SVD y embeddings.

## Experimentación iii (notebook: celdas #24 a #27): SVD (temas) — vertical `rd`

**Qué se probó.** SVD/LSA sobre el TF-IDF de los contratos del pasado: cada texto se representa por "temas"
(combinaciones de palabras que aparecen juntas) en lugar de palabras exactas. Variantes: SVD 100 y SVD 300 en lugar
del TF-IDF, y TF-IDF + 0,5 × SVD 100 (peso fijado de antemano). Misma regla de decisión que en la Experimentación ii.

**Resultado en selección (2023, 130 licitaciones)**

| Variante | Mejor alpha | MRR | Δ vs TF-IDF base | IC95 del Δ | Mejoran / empeoran / iguales |
|---|---|---|---|---|---|
| TF-IDF base | 1 | 0,143 | — | — | — |
| SVD 100 | 0 (no usarla) | 0,125 | −0,018 | (−0,048; +0,010) | 50 / 35 / 45 |
| SVD 300 | 0 (no usarla) | 0,125 | −0,018 | (−0,048; +0,010) | 50 / 35 / 45 |
| TF-IDF + SVD 100 | 1,5 | 0,141 | −0,002 | (−0,015; +0,013) | 26 / 35 / 69 |

**Lectura**
- SVD sola empeora al B0 de forma significativa: con SVD 100 y alpha = 1, MRR 0,108 e IC95 del Δ contra el B0
  de −0,031 a −0,005. Por eso su mejor alpha es 0. Con 300 temas empeora menos que con 100.
- Sumada al TF-IDF no agrega nada.
- Se mantiene el TF-IDF base; no se mide nada nuevo en prueba (sigue valiendo la Experimentación i).
- Conclusión: los temas son demasiado generales y pierden las palabras raras que identifican a la empresa correcta.
  Junto con la Experimentación ii: lo que sirve del texto son las coincidencias de palabras específicas.


## Experimentación v: embeddings (all-MiniLM-L6-v2)

Un vector por contrato (modelo all-MiniLM-L6-v2, 384 dimensiones, hasta 256 tokens),
promedio por empresa con contratos anteriores al corte de cada semestre. Se probaron
embeddings solos y mezclados mitad y mitad con TF-IDF, sobre B0.

| alpha | B0+TF-IDF (base) | B0+embeddings | B0+mezcla TF-IDF y emb |
|------:|-----------------:|--------------:|-----------------------:|
| 0.00 | 0.125 | 0.125 | 0.125 |
| 0.05 | 0.124 | 0.128 | 0.126 |
| 0.10 | 0.126 | 0.130 | 0.129 |
| 0.25 | 0.127 | 0.133 | 0.133 |
| 0.50 | 0.129 | 0.134 | 0.135 |
| 0.75 | 0.133 | 0.135 | 0.143 |
| 1.00 | 0.142 | 0.138 | 0.145 |
| 1.50 | 0.137 | 0.137 | 0.159 |
| 2.00 | 0.138 | 0.136 | 0.153 |

- Selección (2023): mejor = mezcla con alpha 1.5 (MRR 0.159). Δ vs base = +0.016,
  IC95 [+0.006; +0.030]; 55 mejoran, 19 empeoran, 56 iguales. La regla fijada indica pasar a prueba.
- Prueba (2024S1): base 0.112, nueva 0.110. Δ = -0.002, IC95 [-0.024; +0.022];
  12 mejoran, 11 empeoran, 34 iguales. No hay mejora medible.
- Lectura: la ventaja de selección es en parte optimista (se eligió la mejor de 18
  combinaciones sobre los mismos datos con que se mide). Con 57 licitaciones de prueba
  solo se detectan diferencias grandes. En rd, los embeddings no aportan una mejora medible.


### Experimentación v: desglose por S_inc y S_new

S_inc = licitaciones donde B0 pone un ganador en su top-10; S_new = el resto.
Δ MRR es contra B0, con IC95 por bootstrap pareado.

| etapa | método | conjunto | N | MRR | B0 MRR | Δ MRR | IC95 | Hit@5 | B0 Hit@5 |
|---|---|---|--:|--:|--:|--:|---|--:|--:|
| selección 2023 | B0+TF-IDF | todas | 130 | 0.142 | 0.125 | +0.017 | (-0.010, +0.047) | 0.269 | 0.246 |
| selección 2023 | B0+TF-IDF | S_inc | 41 | 0.346 | 0.361 | -0.015 | (-0.074, +0.053) | 0.659 | 0.780 |
| selección 2023 | B0+TF-IDF | S_new | 89 | 0.048 | 0.016 | +0.032 | (+0.007, +0.064) | 0.090 | 0.000 |
| selección 2023 | mezcla TF-IDF y emb | todas | 130 | 0.159 | 0.125 | +0.034 | (+0.002, +0.070) | 0.285 | 0.246 |
| selección 2023 | mezcla TF-IDF y emb | S_inc | 41 | 0.354 | 0.361 | -0.007 | (-0.065, +0.063) | 0.683 | 0.780 |
| selección 2023 | mezcla TF-IDF y emb | S_new | 89 | 0.068 | 0.016 | +0.052 | (+0.017, +0.094) | 0.101 | 0.000 |
| prueba 2024S1 | B0+TF-IDF | todas | 57 | 0.112 | 0.091 | +0.021 | (-0.008, +0.056) | 0.228 | 0.228 |
| prueba 2024S1 | B0+TF-IDF | S_inc | 16 | 0.304 | 0.294 | +0.010 | (-0.079, +0.128) | 0.688 | 0.812 |
| prueba 2024S1 | B0+TF-IDF | S_new | 41 | 0.037 | 0.011 | +0.026 | (+0.005, +0.051) | 0.049 | 0.000 |
| prueba 2024S1 | mezcla TF-IDF y emb | todas | 57 | 0.110 | 0.091 | +0.020 | (-0.023, +0.068) | 0.211 | 0.228 |
| prueba 2024S1 | mezcla TF-IDF y emb | S_inc | 16 | 0.301 | 0.294 | +0.007 | (-0.130, +0.181) | 0.562 | 0.812 |
| prueba 2024S1 | mezcla TF-IDF y emb | S_new | 41 | 0.036 | 0.011 | +0.024 | (+0.005, +0.049) | 0.073 | 0.000 |

- El texto mejora en S_new de forma consistente (IC95 positivo en selección y en prueba).
- En S_inc el MRR no cambia de forma medible, pero el Hit@5 baja (en prueba N = 16).
- La mezcla con embeddings no supera a B0+TF-IDF en S_new en prueba (+0.024 contra +0.026).
- Los grupos se definen con el resultado de B0: la mejora en S_new es en parte esperable
  y no se puede usar el grupo para decidir al predecir.


## Experimentación vi: B0 y B0+TF-IDF en las cinco verticales (experimento.py)

Mismo método que la Experimentación i, pero en rd, d, it, r y j: selección 2023, prueba 2024S1, alpha elegido
en selección por MRR. Se corre con `python experimento.py <vertical>` (necesita `b0_<vertical>.csv`, que genera `b0.py`).

B0 sobre todas las licitaciones de cada vertical:

| vertical | licitaciones | MRR | Hit@5 | Hit@1 | Hit@100 | puesto mediano |
|---|--:|--:|--:|--:|--:|--:|
| rd | 328 | 0.125 | 0.271 | 0.079 | 0.628 | 9 |
| d | 819 | 0.092 | 0.112 | 0.056 | 0.410 | 15 |
| it | 1792 | 0.081 | 0.125 | 0.035 | 0.410 | 15 |
| r | 2180 | 0.175 | 0.239 | 0.118 | 0.525 | 7 |
| j | 4504 | 0.164 | 0.222 | 0.103 | 0.543 | 9 |

B0+TF-IDF en prueba (2024S1), con el alpha elegido en selección:

| vertical | alpha | N | MRR | B0 MRR | Δ MRR | IC95 |
|---|--:|--:|--:|--:|--:|---|
| rd | 1.0 | 57 | 0.112 | 0.091 | +0.021 | (-0.008, +0.056) |
| d | 1.5 | 161 | 0.203 | 0.090 | +0.113 | (+0.065, +0.164) |
| it | 1.0 | 319 | 0.111 | 0.088 | +0.023 | (+0.006, +0.041) |
| r | 1.0 | 468 | 0.237 | 0.194 | +0.043 | (+0.026, +0.060) |
| j | 0.75 | 995 | 0.151 | 0.136 | +0.015 | (+0.008, +0.023) |

- El texto mejora a B0 de forma significativa en d, it, r y j. En rd el signo es el mismo pero el IC95 cruza 0 (pocas licitaciones).
- La mejora está en S_new (B0 no ve al ganador): Δ MRR rd +0.026, d +0.119, it +0.029, r +0.048, j +0.016, todos con IC95 positivo.
- En S_inc el MRR no cambia de forma significativa, pero el Hit@5 baja en rd (0.812 a 0.688), d (0.769 a 0.731), it (0.823 a 0.581) y r (0.809 a 0.757); en j queda igual.
- Un alpha global de 1 queda en el mejor valor o a menos de 0.01 de él en las cinco verticales; en j el alpha 2.0 queda significativamente por debajo de B0 en selección (Δ -0.042, IC95 -0.055 a -0.030) y en prueba (Δ -0.021, IC95 -0.037 a -0.004), y el 1.5 no se distingue de B0 (selección -0.009, IC95 -0.019 a +0.003; prueba +0.003, IC95 -0.011 a +0.017).

## Experimentación v (continuación): embeddings en las cinco verticales (experimento_emb.py)

Modelo all-MiniLM-L6-v2 (256 tokens), un vector por contrato y promedio por empresa con corte cronológico.
La mezcla suma la similitud TF-IDF y 0.5 veces la de embeddings (cada una normalizada por su máximo). Se corre con
`python experimento_emb.py <verticales>` y necesita GPU y sentence-transformers.

Mezcla (mejor alpha en selección) en prueba, contra la base TF-IDF y contra B0:

| vertical | alpha | MRR mezcla | MRR base TF-IDF | MRR B0 | Δ vs base TF-IDF | Δ vs B0 |
|---|--:|--:|--:|--:|---|---|
| rd | 1.5 | 0.110 | 0.112 | 0.091 | -0.002 (-0.024, +0.022) | +0.020 (-0.023, +0.068) |
| d | 2.0 | 0.212 | 0.203 | 0.090 | +0.008 (-0.005, +0.021) | +0.122 (+0.075, +0.171) |
| it | 1.5 | 0.138 | 0.111 | 0.088 | +0.027 (+0.014, +0.041) | +0.050 (+0.026, +0.076) |
| r | 1.5 | 0.270 | 0.237 | 0.194 | +0.033 (+0.021, +0.047) | +0.076 (+0.053, +0.099) |
| j | 1.5 | 0.176 | 0.151 | 0.136 | +0.025 (+0.017, +0.032) | +0.040 (+0.028, +0.052) |

- La mezcla mejora de forma significativa a la base TF-IDF en it, r y j (en selección y en prueba). En d mejora en selección (+0.024, IC95 +0.013 a +0.036) pero en prueba no alcanza significancia (+0.008, IC95 -0.005 a +0.021); en rd mejora en selección (+0.016, IC95 +0.006 a +0.030) pero en prueba no se distingue (-0.002, IC95 -0.024 a +0.022), con solo 57 licitaciones.
- En r y j mejora también S_inc contra B0 (r +0.066, IC95 +0.020 a +0.115; j +0.036, IC95 +0.001 a +0.075). En it el Hit@5 de S_inc baja (0.823 a 0.516).
- El mejor alpha de la mezcla es 1.5 en rd, it, r y j (2.0 en d, borde de la grilla): un alpha global de 1.5 es el mejor en it, r y j, y en d queda a 0.002 del mejor en selección (0.180 contra 0.182).
- Los embeddings solos tienen su mejor alpha en el borde de la grilla (2.0) en d, it, r y j: la grilla era corta. Pendiente: grilla hasta 6 y un modelo más potente (all-mpnet-base-v2, 384 tokens).
- Conclusión provisoria: con licitaciones suficientes (it, r, j) los embeddings suman sobre TF-IDF; en rd (57 licitaciones de prueba) no se podía ver.


## Experimentación vii: grilla extendida y comparación de modelos de embeddings (experimento_emb.py)

Se repite la mezcla TF-IDF + embeddings con alphas 0, 0.5, 1, 1.5, 2, 3, 4 y 6 (`--alphas`), con all-MiniLM-L6-v2 (256 tokens) y con
all-mpnet-base-v2 (768 dimensiones, 384 tokens, `--modelo all-mpnet-base-v2 --largo 384`). Mismo corte cronológico y misma regla de decisión.

Mejor alpha en selección (2023) y su MRR, MiniLM, grilla extendida:

| vertical | TF-IDF solo | embeddings solos | mezcla |
|---|---|---|---|
| rd | 1.0 (0.142) | 1.0 (0.138) | 1.5 (0.159) |
| d | 1.5 (0.158) | 3.0 (0.179) | 2.0 (0.182) |
| it | 1.0 (0.120) | 2.0 (0.132) | 1.5 (0.148) |
| r | 1.0 (0.232) | 2.0 (0.249) | 1.5 (0.251) |
| j | 1.0 (0.206) | 2.0 (0.220) | 1.5 (0.227) |

Mezcla en prueba (2024S1), con el alpha elegido en selección, contra la base TF-IDF:

| vertical | MiniLM: alpha | MRR | Δ vs base TF-IDF | mpnet: alpha | MRR | Δ vs base TF-IDF |
|---|--:|--:|---|--:|--:|---|
| rd | 1.5 | 0.110 | -0.002 (-0.024, +0.022) | 1.5 | 0.112 | -0.000 (-0.022, +0.023) |
| d | 2.0 | 0.212 | +0.008 (-0.005, +0.021) | 3.0 | 0.222 | +0.019 (+0.005, +0.034) |
| it | 1.5 | 0.138 | +0.027 (+0.014, +0.041) | 1.5 | 0.140 | +0.029 (+0.017, +0.041) |
| r | 1.5 | 0.270 | +0.033 (+0.021, +0.047) | 1.5 | 0.266 | +0.029 (+0.018, +0.040) |
| j | 1.5 | 0.176 | +0.022 (+0.016, +0.029) | 1.5 | 0.174 | +0.021 (+0.015, +0.026) |

- Con la grilla extendida los embeddings solos tienen su máximo dentro de la grilla, no en el borde. La mezcla mantiene su mejor alpha en 1.5 (2.0 con MiniLM y 3.0 con mpnet en d).
- El mejor MRR de la mezcla en selección es casi igual con los dos modelos (diferencia de 0.004 o menos) y en prueba la diferencia es de 0.010 o menos, con signo cambiante. mpnet no supera a MiniLM; el método usa MiniLM.
- mpnet tarda entre 1.3 y 1.8 veces más que la primera corrida de MiniLM (que también calculó los vectores): rd 43 s contra 78 s, d 92 contra 164, it 164 contra 284, r 377 contra 658, j 298 contra 381.
- Un alpha global de 1.5 queda a 0.002 (MiniLM) o 0.004 (mpnet) del mejor en selección en d, y es el mejor en las otras cuatro verticales.
- Hit@5 de S_inc con la mezcla (MiniLM) contra B0: rd 0.562 contra 0.812, d 0.769 contra 0.769, it 0.516 contra 0.823, r 0.763 contra 0.809, j 0.678 contra 0.690. Los grupos S_inc son chicos en rd (16 licitaciones) y d (26).
- Con esta grilla la base TF-IDF de j queda en alpha 1.0 (empata con 0.75 en selección), por eso el Δ de j contra la base es +0.022 y no +0.025 como en la Experimentación v.
