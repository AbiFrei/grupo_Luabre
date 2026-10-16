# Y-Hat: ¿quién gana la licitación?

Cada licitación es un documento en el que un organismo describe algo que necesita comprar.
La tarea: **dada una licitación, rankear las empresas candidatas por probabilidad de
ganarla**, usando sólo lo que se sabía cuando se publicó.

La pregunta de fondo no es sólo acertar. Una buena parte se resuelve sin leer nada: las
empresas que ya ganan mucho en esa oficina y en ese rubro suelen volver a ganar. Eso es el
**B0**, que viene dado. **Lo que interesa es cuánto agrega el texto por encima del B0.**

## Cómo trabajar

Cada grupo trabaja en **su propio repo, privado**. No se crean ramas ni se hacen pull
requests acá: este repo es público y lo ven todos los grupos.

1. Crear un repo **vacío y privado** en GitHub (por ejemplo, `grupo-x`). No se puede usar un
   fork: los forks de un repo público son públicos.
2. Clonar éste, dejarlo como `upstream` y apuntar `origin` al repo propio:

   ```
   git clone https://github.com/bravoivanagustin/ranking-licitaciones.git grupo-x
   cd grupo-x
   git remote rename origin upstream
   git remote add origin git@github.com:<usuario>/grupo-x.git
   git push -u origin main
   ```

3. Trabajar ahí con las ramas y la organización que quieran.
4. Cuando se publique algo nuevo acá (una corrección de `evaluar.py` o de los datos),
   traerlo con:

   ```
   git pull upstream main
   ```

## Cómo empezar

```
uv sync          # instala Python (3.10+) y las dependencias en .venv/
```

Los datos no están en el repo: se bajan del release
[**`datos-v1`**](https://github.com/bravoivanagustin/ranking-licitaciones/releases/tag/datos-v1)
(`data.zip`, ~175 MB) y se descomprimen en la raíz del repo, de modo que quede
`data/<vertical>/`. Desde la raíz del repo:

```
curl -LO https://github.com/bravoivanagustin/ranking-licitaciones/releases/download/datos-v1/data.zip
unzip data.zip
```

(o con el CLI de GitHub: `gh release download datos-v1 -R bravoivanagustin/ranking-licitaciones`;
o desde el link, en "Assets"). En Windows, desde PowerShell o cmd (no desde Git Bash, cuyo
`tar` no abre zips): `tar -xf data.zip`, o `Expand-Archive data.zip .` en
PowerShell.

## Los datos

Hay cinco verticales, cada una en su carpeta, con el mismo formato:

| carpeta | rubro |
|---|---|
| `rd` | investigación y desarrollo |
| `j` | mantenimiento y reparación de equipos |
| `d` | IT y telecomunicaciones |
| `r` | servicios profesionales |
| `it` | productos informáticos |

En cada una, cuatro tablas (el `README.md` de cada carpeta tiene el detalle):

| tabla | una fila por | qué es |
|---|---|---|
| `licitaciones.parquet` | licitación | el input: `id`, `t` (publicación), `title`, `description`, `psc`, `naics`, `office`, `set_aside` |
| `ganadores.parquet` | (licitación, empresa) | el label: `id`, `parent_uei` |
| `empresas.parquet` | (empresa, semestre) | las candidatas y qué se sabía de ellas al empezar el semestre |
| `historial.parquet` | contrato | los contratos anteriores, con fecha, rubro, oficina y descripción |

Y en la raíz de `data/`, tres tablas de códigos comunes a las cinco verticales, para
pasar de un código a su significado (otra fuente de texto):

| tabla | columnas | ejemplo |
|---|---|---|
| `psc.parquet` | `codigo`, `descripcion` | `AN42` → HEALTH R&D SERVICES; HEALTH CARE - OTHER; APPLIED RESEARCH |
| `naics.parquet` | `codigo`, `descripcion`, `version` | `541715` → Research and Development in the Physical, Engineering, and Life Sciences (…) |
| `office.parquet` | `codigo`, `oficina`, `subagencia`, `departamento` | `N62645` → NAVAL MEDICAL LOGISTICS COMMAND · DEPT OF THE NAVY · DEPT OF DEFENSE |

- `psc` (qué se compra) y `naics` (de qué industria es el trabajo) se cruzan con las columnas
  del mismo nombre de `licitaciones` e `historial`; `office` (quién compra), con `office`.
- `naics` junta las versiones 2002 a 2022 de la clasificación, porque el historial las mezcla
  (`511210` y `513210` son los dos "Software Publishers"); `version` dice de cuál sale la
  descripción. Algunos avisos traen el NAICS truncado (`541`): también están.
- Todas las oficinas de las licitaciones tienen nombre. Algunas que sólo aparecen en el
  `historial` tienen subagencia y departamento pero no nombre de oficina.

**Las candidatas de una licitación son las filas de `empresas` de su semestre**
(`2024S1` = enero–junio, `2024S2` = julio–diciembre): las empresas que ya habían ganado
algún contrato del rubro antes de que empezara. `evaluar.py` rechaza un ranking que incluya
cualquier otra empresa.

**Todo es anterior al 1/7/2024.** Los datos no vienen divididos en train y validación:
cada grupo arma la suya. Conviene que sea temporal: entrenar con lo más viejo y validar con
lo más nuevo. Por ejemplo, validar con el último semestre:

```python
lic = pd.read_parquet("data/j/licitaciones.parquet")
gan = pd.read_parquet("data/j/ganadores.parquet")
val = lic[lic.t >= "2024-01-01"]
gan[gan.id.isin(val.id)].to_parquet("mi_val_j.parquet")
```

**Ojo con el futuro.** Para una licitación publicada en `t`, sólo vale usar lo anterior a
`t`: las filas de `empresas` de su semestre ya lo cumplen; del `historial`, sólo las filas
con `fecha < t`. Un modelo que en validación usa el historial completo se ve muy bien y
no sirve de nada con licitaciones nuevas.

## Las métricas

`evaluar.py` recibe un CSV por vertical con el ranking de **hasta 100 empresas** por
licitación (pueden ser menos; un ganador que no está en la lista cuenta 0):

```
id,parent_uei,puesto
FA865124R0012,ABC123DEF456,1
FA865124R0012,XYZ987GHI654,2
```

**La métrica de referencia es el MRR**: 1 / puesto del ganador, promediado sobre las
licitaciones (0 si el ganador no está en el top 100). Lo que se compara entre grupos es
**cuánto mejora el MRR sobre el del B0**, en las mismas licitaciones y con su intervalo.

Algunas licitaciones tienen varios ganadores (contratos adjudicados a más de una empresa;
pasa en ~1/3 de las de `rd` y en menos del 4% en el resto). Ahí cada ganador se puntúa con
1/puesto **sin contar a los otros ganadores que tiene arriba**, y la licitación vale el
promedio. Con ganadores en los puestos 1, 2 y 50: puestos 1, 1 y 48, y
(1 + 1 + 1/48) / 3 = 0,67. Con un solo ganador es el MRR de siempre.

Se reportan además `Hit@5` (algún ganador en el top 5) y `Hit@1`, y todo por separado para
las licitaciones donde el B0 ya pone a un ganador en su top 10 (`S_inc`) y para el resto
(`S_new`), que es donde el texto tiene que hacer todo el trabajo.

El MRR se decide casi todo en los primeros puestos: subir un ganador del puesto 80 al 30
suma 0,02, del 5 al 1 suma 0,8. Para ver el progreso intermedio, `evaluar.py` muestra
también tres diagnósticos (no cuentan para la comparación):

- `Hit@100`: en qué fracción de las licitaciones aparece algún ganador en la lista.
- `Recall@100`: qué fracción de los ganadores aparece en la lista, promediada por licitación.
  Con un ganador es igual a `Hit@100`; con varios, cuenta a cada uno.
- `puesto mediano`: el puesto del mejor ganador, entre las licitaciones donde aparece. Se
  lee junto con el recall: una lista más corta lo baja aunque el modelo no sea mejor.

Para evaluar un ranking sobre la validación propia:

```
uv run evaluar.py mi_ranking_j.csv --datos data/j --respuestas mi_val_j.parquet --b0 b0_j.csv
```

## El B0

El B0 rankea a las candidatas **sólo por quién ganó antes**, sin leer ningún texto: las
empresas que ya ganaron contratos parecidos, sobre todo en la misma oficina y hace poco,
van arriba.

### Cómo se calcula

Para una licitación `q` publicada en `t_q`, cada contrato previo `a` del `historial` (de la
empresa `e_a`, firmado en `t_a < t_q`) pesa según su antigüedad:

```
w(a) = exp( −(t_q − t_a) / τ ),      τ = 2,5 años
```

Hay cinco **niveles**, cada uno con un criterio de qué contratos previos "se parecen" a la
licitación, y un peso fijo:

| nivel | el contrato previo tiene… | peso λ |
|---|---|---|
| PSC + oficina | el mismo `psc` y la misma `office` | 0,40 |
| PSC | el mismo `psc` | 0,24 |
| PSC + set-aside | el mismo `psc` y el mismo tipo de `set_aside` | 0,20 |
| NAICS | el mismo `naics` | 0,12 |
| global | (cualquiera) | 0,04 |

En cada nivel, la probabilidad de una empresa es la parte del peso que se lleva entre los
contratos previos que cumplen el criterio:

```
               Σ  w(a)   sobre los contratos del nivel ganados por e
P_nivel(e) = ─────────────────────────────────────────────────────────     (0 si no hay ninguno)
               Σ  w(a)   sobre todos los contratos del nivel
```

y el B0 es la mezcla:

```
p_b0(e) = 0,40 · P_psc+oficina(e) + 0,24 · P_psc(e) + 0,20 · P_psc+set-aside(e)
        + 0,12 · P_naics(e)        + 0,04 · P_global(e)
```

Las candidatas se ordenan por `p_b0` de mayor a menor. Detalles:

- **Sólo cuenta lo anterior a `t_q`** (estrictamente): el historial está completo hasta el
  corte, pero cada licitación ve sólo su pasado.
- **El PSC es el código completo** (`DA01`), no la familia.
- **El set-aside se agrupa** en 8A, SDVOSB, WOSB, HUBZONE, VOSB, SMALL, NONE y OTRO, porque
  la licitación y el historial lo escriben distinto.
- **Si un nivel no tiene contratos previos, aporta 0** y su peso no se redistribuye.
- **Los empates** (sobre todo las muchas empresas con `p_b0 = 0`, que nunca ganaron nada
  parecido) se ordenan al azar, con una semilla que sale del `id` de la licitación.
- **No se entrena nada**: `τ` y los pesos están fijos.

El nivel que hace el trabajo es PSC + oficina: cada oficina tiende a contratar con las
mismas empresas.

### Cómo se usa

`b0.py` calcula el baseline. Es determinístico: el mismo comando da lo mismo en cualquier
máquina.

```
uv run b0.py data/j --salida b0_j.csv
```

Como módulo da `p_b0` para todas las candidatas, para combinarlo con un modelo propio:

```python
import b0
prior = b0.B0(pd.read_parquet("data/j/historial.parquet"))
p = prior.p_b0(licitacion, empresas)    # pd.Series indexada por parent_uei
```

Lo que tiene que dar sobre todo `data/` (si da otra cosa, algo cambió):

| vertical | licitaciones | MRR | Hit@5 | Hit@1 |
|---|---|---|---|---|
| `rd` | 328 | 0,125 | 0,271 | 0,079 |
| `d` | 819 | 0,092 | 0,112 | 0,056 |
| `j` | 4.504 | 0,164 | 0,222 | 0,103 |
| `r` | 2.180 | 0,175 | 0,239 | 0,118 |
| `it` | 1.792 | 0,081 | 0,125 | 0,035 |

```
uv run b0.py data/j --salida b0_j.csv
uv run evaluar.py b0_j.csv --datos data/j
```

