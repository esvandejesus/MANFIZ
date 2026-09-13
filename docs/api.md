# API de MANFIZ 0.1.0

## Datos y unidades

Todas las llamadas públicas reciben datos y cotas en unidades físicas. `X`
tiene forma `(N,d)` e `y` forma `(N,m)`. Una `y` unidimensional se interpreta
como una única salida. La salida de predicción siempre tiene forma `(N,m)`.
Cada elemento de `groups` identifica el experimento al que pertenece la fila.

`Standardizer.fit` calcula media y desviación muestral (`ddof=1`) usando
exclusivamente entrenamiento. Una columna constante recibe escala 1; una
variable de premisa constante se rechaza porque no define una partición útil.

## Constructor y configuración

```python
from manfiz import MANFIZ, FitConfig, PremiseConfig, ReductionConfig
cfg = FitConfig(
    premise=PremiseConfig(max_iter=6000, max_evaluations=10000),
    reduction=ReductionConfig(q_target=100, q_max=1000),
    batch_size=16, max_buffer_size=64, pe_ratio_min=1e-3,
    dominant_weight_min=0.20,
)
model = MANFIZ((4, 6, 8), n_memberships=(2, 2, 2), config=cfg)
```

`n_memberships` puede ser escalar o una cantidad por variable de premisa.
`weight` es una matriz simétrica definida positiva de tamaño `(d+1,d+1)`;
el valor predeterminado es la identidad. La configuración se copia al construir
el estimador, por lo que modificar `cfg` después no modifica el modelo.

| Parámetro | Valor predeterminado | Significado |
|---|---:|---|
| `premise.max_iter` / `max_evaluations` | 2000 / 3500 | Presupuestos Nelder-Mead; detiene el primero alcanzado |
| `premise.x_tolerance` / `objective_tolerance` | 1e-4 / 1e-6 | Tolerancias del optimizador |
| `premise.ridge` / `penalty` | 1e-8 / 1e-4 | Regularización lineal y penalización de premisas |
| `premise.optimize` | True | False conserva la malla inicial; se informa explícitamente |
| `reduction.q_target` / `q_max` | 100 / 1000 | Intervalo de búsqueda del número de generadores |
| `reduction.q_trigger` | None | None usa `q_max` como disparador |
| `reduction.max_inflation` | 1.10 | Umbral de inflación del radio F ponderado |
| `reduction.max_cycle_ratio` | 1.005 | Umbral del cociente posterior/propagado |
| `batch_size` / `max_buffer_size` | 16 / 64 | Filas por lote y tamaño objetivo del búfer |
| `pe_ratio_min` | 1e-3 | Mínimo cociente singular menor/mayor del lote |
| `dominant_weight_min` | 0.20 | Mínima activación de la regla asignada |
| `prior_shape_floor` / `prior_radius_floor` | 0.02 / 1e-7 | Piso relativo de forma y piso de radio normalizado |
| `prior_safety` / `regression_bound_safety` | 1.05 / 1.05 | Márgenes del prior y de la cota total de entrenamiento |
| `min_samples_per_group` | 100 | Muestras mínimas de cada reajuste por experimento |
| `parameter_drift` | 0 | Radio isotrópico añadido antes de cada actualización local aceptada |
| `verify_training_witness` | True | Verifica pertenencia a los zonotopos finales si drift=0 |

Se requiere `d+1 <= batch_size <= max_buffer_size` y
`d+1 <= q_target <= q_trigger <= q_max`. El tratamiento de deriva es por
actualización local aceptada, como en el flujo original; no es un modelo
general de variación de todos los parámetros en cada instante de tiempo.

## Métodos

| Método | Función |
|---|---|
| `fit(X,y,groups=...,sensor_bound=...)` | Entrena las dos etapas desde cero y devuelve `self` |
| `fit_nominal(X,y)` | Inicializa y entrena premisas y consecuentes nominales |
| `fit_zonotopes(X,y,groups=...,sensor_bound=...)` | Recalibra prior y cota total y reinicia los zonotopos; exige exactamente los datos usados por `fit_nominal` |
| `predict(X,nominal=False)` | Centro posterior; `nominal=True` devuelve el centro MANFIS |
| `features(X)` | Regresores afines normalizados y pesos normalizados |
| `coefficients(nominal=False)` | Tensor `(n_rules,d+1,m)` de coeficientes normalizados |
| `calibrate_validation(X,y,safety=1.05)` | Congela caja residual por salida sobre validación |
| `calibrate_noise(clean_X,clean_y,sensor_bound=...,regressor_bound=...,safety=1.10)` | Congela la extensión de ruido sobre una calibración limpia independiente |
| `predict_interval(X,mode=...,chunk_size=2048)` | Devuelve `IntervalPrediction` con centro, radio, componentes, `lower` y `upper` |
| `save(path)` / `MANFIZ.load(path)` | Persistencia NPZ versionada sin pickle ni objetos MATLAB |

`calibrate_noise` mantiene como mínimo la caja residual de validación si está
disponible. `minimum_model_defect=` permite suministrar otro piso no negativo;
su validez fuera de las muestras de calibración es una hipótesis que debe
justificarse. Calibrar de nuevo la validación invalida la calibración de ruido.
Reajustar los consecuentes invalida ambas calibraciones, para evitar reutilizar
cotas calculadas para un centro o zonotopo anterior.

`fit_report_["premise"]` registra los parámetros iniciales/finales, objetivo,
historial y estado de terminación. `success=False` no se transforma en una
afirmación de convergencia. `outputs_` contiene centros, generadores y trazas
por salida; `feasibility_` contiene el testigo común y residuos de cada LP.
Estos atributos con sufijo `_` son diagnósticos avanzados, no datos de entrada.

## NARX y simulación libre

`NARXSpec(output_lags=(1,2),input_lags=(1,2),discard=30)` construye columnas por
salida y después por entrada, con los retardos en el orden declarado. Para el
benchmark: `[y1(k-1),y1(k-2),y2(k-1),y2(k-2),u1(k-1),u1(k-2),u2(k-1),u2(k-2),u3(k-1),u3(k-2)]`.

`transform_runs([(u1,y1),...])` no mezcla retardos entre experimentos.
`free_run(model,u,initial_y)` usa solo `initial_y[:start]` y realimenta luego
sus propios centros. Produce una simulación de centros, no una propagación
recursiva de intervalos. Las cotas del modo `noise` son de un paso con retardos
medidos; no se deben interpretar como cotas de simulación libre.

## Geometría y comparación opcional

`Zonotope(center,generators)` ofrece `interval`, `linear_map`, `minkowski_sum`
y `reduce`. Los módulos `zonotopes`, `calibration` y `recursion` exponen el
núcleo matemático para inspección y experimentación.

`manfiz.baselines.SharedFeatureQuantile` usa exactamente el diseño difuso
congelado. Requiere la dependencia opcional scikit-learn. La calibración por
cuantil no se presenta como una garantía determinista, ni como una garantía
conformal para series autocorrelacionadas sin justificar intercambiabilidad.
