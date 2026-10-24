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
