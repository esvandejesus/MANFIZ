# MANFIZ Toolbox

Identificación neurodifusa multisalida con consecuentes zonotópicos, implementada
en Python. `MANFIZ.fit()` **entrena desde cero las premisas compartidas, los
consecuentes nominales y los zonotopos de cada salida**. No requiere MATLAB,
Octave, Fuzzy Logic Toolbox ni archivos `.mat`.

Esta versión contiene tres modelos recién entrenados, datos reproducibles,
comparaciones, figuras y 21 pruebas automatizadas. El alcance es identificación
y predicción por intervalos; no incluye esquemas de detección de fallas.

## Instalar

Requiere Python 3.10 o posterior. Desde la carpeta del proyecto:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[all]"
```

El núcleo usa NumPy y SciPy. `[plots]` añade Matplotlib y `[comparisons]` añade
scikit-learn. El wheel incluido en `dist/` permite instalar el núcleo sin el
árbol de fuentes; los modelos, ejemplos y resultados están en el ZIP del proyecto.

## Entrenar y predecir

```python
import numpy as np
from manfiz import MANFIZ, NARXSpec
from manfiz.benchmarks import make_training_runs

spec = NARXSpec()
runs = make_training_runs(system=1)
train = spec.transform_runs([(r.u, r.y) for r in runs[:4]])
validation = runs[4].regression(spec)
test = runs[5].regression(spec)

model = MANFIZ(premise_columns=(4, 6, 8), n_memberships=2)
model.fit(train.X, train.y, groups=train.groups,
          sensor_bound=np.sqrt(3) * 0.01)
model.calibrate_validation(validation.X, validation.y)

interval = model.predict_interval(test.X, mode="validation")
center, lower, upper = interval.center, interval.lower, interval.upper
model.save("results/my_model.npz")
loaded = MANFIZ.load("results/my_model.npz")
```

`premise_columns` usa índices Python desde cero sobre **X**, la matriz de
regresores. `sensor_bound` es una amplitud máxima determinista en unidades
físicas. Una desviación estándar no es, por sí sola, una cota máxima.

Para datos propios: `examples/custom_data.py` recibe archivos NPZ con `u` de
forma `(N, n_inputs)` e `y` de forma `(N, n_outputs)`, uno por experimento.
Se necesitan al menos dos experimentos de entrenamiento con suficientes muestras
para calibrar la forma del prior. Se conserva la separación entre experimentos
al construir los retardos. El producto cartesiano de membresías crece
exponencialmente con el número de variables de premisa.

## Intervalos que incluyen ruido

Hay tres modos explícitos:

| Modo | Radio añadido al centro MANFIZ |
|---|---|
| `parameters` | Incertidumbre de los consecuentes |
| `validation` | Radio paramétrico + caja residual de validación |
| `noise` | Radio paramétrico + defecto limpio + ruido actual + propagación por los retardos |

Para `noise`, hay que congelar antes una envolvente limpia y cotas deterministas:

```python
# clean_cal es RegressionData obtenido de trayectorias sin ruido de calibración.
# Debe estar separado de los datos de confirmación.
b = np.sqrt(3) * 0.02
bx = spec.regressor_noise_bounds(3, np.array([b, b]))
model.calibrate_noise(clean_cal.X, clean_cal.y,
                      sensor_bound=b, regressor_bound=bx)
interval = model.predict_interval(test.X, mode="noise")
```

Este bloque ilustra la API; el experimento completo que genera `clean_cal` y
mantiene la separación de datos está implementado en `manfiz-benchmark`.
Las variables de premisa deben estar libres de ruido para esta extensión.
En datos reales, una trayectoria limpia o una envolvente limpia válida requiere
justificación independiente; la toolbox no la deduce de las mediciones ruidosas.

**La cobertura de todos los ruidos acotados es condicional a que la envolvente
limpia sea válida.** Un estudio finito sin violaciones no demuestra cobertura
universal de trayectorias futuras. Un ruido gaussiano sin recorte tiene soporte
no acotado y no admite una cota determinista finita que incluya todos sus valores.

## Reproducir los resultados incluidos

El comando siguiente reproduce el presupuesto utilizado en la entrega. Use una
carpeta nueva: el programa protege los experimentos existentes.

```bash
# Linux/macOS: fija los hilos para reducir la variación en los tiempos.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 manfiz-benchmark \
  --output results/my_run --max-iter 6000 --max-evaluations 10000

# Después, sobre los modelos recién entrenados:
python examples/budget_sensitivity.py --results results/my_run
python examples/quantile_comparison.py --results results/my_run
python tools/verify_saved_results.py --results results/my_run
python -m unittest discover -s tests -v
```

En Windows, establezca `$env:OPENBLAS_NUM_THREADS="1"` y
`$env:OMP_NUM_THREADS="1"`, y ejecute el comando `manfiz-benchmark` en una línea.
El presupuesto predeterminado de la API es 2000/3500; el experimento entregado
lo amplió explícitamente a 6000/10000. Las semillas están en `protocol.json`.

## Resultados nuevos de esta entrega

| Sistema | Iteraciones | Reducción de la función objetivo | Terminación |
|---|---:|---:|---|
| 1 | 6000 | 5,79 % | Límite de iteraciones |
| 2 | 6000 | 24,44 % | Límite de iteraciones |
| 3 | 4286 | 6,44 % | Convergencia numérica |

Se efectuaron 2406 actualizaciones locales de consecuentes entre las seis
salidas. La factibilidad conjunta y la pertenencia del testigo a los 48 zonotopos
finales se verificaron numéricamente. En 540 trayectorias de confirmación hubo
**0 violaciones entre 1.911.600 valores medidos**, con amplitudes 1,00/1,15/1,30
y desviaciones de ruido uniforme 0,005/0,010/0,020. La anchura media pasó de
0,386877 a 0,508028 al añadir la extensión de ruido, un aumento de 31,32 %.
Estos valores corresponden a los instantes con predicción, `k=30,...,1799`.
Los primeros 30 instantes se usan como historia inicial y descarte.

La reducción utilizó la alternativa de mínima violación en 653 de 712 eventos:
el límite de complejidad y la inclusión externa se conservan, pero los umbrales
de inflación/contracción no se garantizan en esos eventos. Los archivos de
resultados registran estas situaciones, no las ocultan.

## Archivos y documentación

| Ruta | Contenido |
|---|---|
| `src/manfiz/` | API y algoritmos reutilizables |
| `examples/` | Entrenamiento, datos propios, presupuestos, cuantiles y simulación libre |
| `tests/` | Verificaciones de geometría, causalidad, entrenamiento, ruido y equivalencia |
| `results/fresh_training/` | Tres modelos nuevos, datos, semillas, trazas, métricas y figuras |
| `results/budget_sensitivity/` | Nueve ajustes de consecuentes con premisas nuevas compartidas |
| `results/quantile_comparison/` | Comparación por cuantiles con las mismas características |
| `docs/mathematics.md` | Ecuaciones, variables, supuestos y límites |
| `docs/api.md` | Parámetros y uso de la API |
| `docs/matlab_mapping.md` | Correspondencia funcional MATLAB/Python |
| `docs/reproducibility.md` | Protocolo, versiones, semillas y comandos |
| `docs/training_report.md` | Informe de hallazgos y correcciones con resultados nuevos |

El código y los resultados están preparados para cargarlos a un repositorio;
no se ha publicado ningún repositorio ni paquete. La licencia de publicación
queda por definir por el autor; véase `NOTICE.md`. `CITATION.cff` contiene la
autoría del manuscrito suministrado sin inventar DOI ni URL.

## English overview

MANFIZ is a native Python research toolbox for shared-premise multi-output
neuro-fuzzy identification with zonotopic consequents. The full `fit` call
initializes and optimizes the premises, refits nominal affine coefficients,
calibrates a joint feasible training bound and fits the local zonotopes. It
supports bounded-noise one-step intervals, saved models and recursive center
simulation. The supplied study is freshly trained, not a replay of MATLAB
parameters. Coverage claims are conditional on a valid clean-output envelope;
see the equations and limitations in `docs/mathematics.md`.
