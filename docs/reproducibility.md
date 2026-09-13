# Reproducción y publicación

## Protocolo ejecutado

Fecha: 13 de septiembre de 2026. Entorno registrado: Python 3.12.14,
NumPy 2.3.5, SciPy 1.17.0, un hilo BLAS/OMP. Se verificó la ejecución local
en ese entorno. La matriz de CI para Python 3.10 y 3.12 está preparada;
no se afirma que se haya ejecutado en GitHub.

Cada sistema usa seis trayectorias nuevas de 1800 muestras: cuatro para
entrenamiento, una para validación y una retenida para evaluación. Tras
descartar 30 muestras por trayectoria se obtienen 7080/1770/1770 filas.
Las fases del protocolo base siguen la fórmula suministrada; el ruido se
genera de nuevo con PCG64. Las fases base son deterministas, por lo que la
trayectoria retenida no es una muestra aleatoria de todo el dominio.

La cota del sensor en entrenamiento es `sqrt(3)*0.01`. El entrenamiento
se inicia con campanas construidas desde el rango de X, con 8 reglas,
18 parámetros de premisa y 11 coeficientes por regla/salida. Se ejecuta
Nelder-Mead con máximos 6000 iteraciones / 10000 evaluaciones. La
normalización se obtiene exclusivamente de las cuatro trayectorias de
entrenamiento. Prior y cota total también usan solo ese conjunto.

Para ruido se fija antes de confirmar:

| Elemento | Valor |
|---|---|
| Calibración limpia | 20 fases por sistema y amplitud; 180 trayectorias |
| Confirmación | 20 fases por sistema, amplitud y ruido; 540 trayectorias |
| Amplitudes | 1.00, 1.15, 1.30 |
| Desviaciones de ruido uniforme | 0.005, 0.010, 0.020 |
| Cota física congelada del sensor | sqrt(3)*0.020 = 0.0346410161514 |
| Piso del defecto limpio | Caja residual de validación |
| Margen de calibración limpia | 1.10 |
| Base de semilla de entrenamiento | 20260913 |
| Base de fases de calibración | 326091300 |
| Base de fases de confirmación | 426091300 |

Para calibración/confirmación se suma `1000*system + replicate` a la base.
El ruido usa la semilla de fases más 100000. Las fases se emparejan entre
amplitudes y el ruido entre intensidades; esa dependencia se conserva y se
declara. Las trayectorias de calibración no se reutilizan como confirmación.
Todas las métricas de cobertura corresponden a muestras k=30,...,1799.

## Reproducción completa

```bash
python -m pip install -e ".[all]"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 manfiz-benchmark \
  --output results/my_run --max-iter 6000 --max-evaluations 10000
python tools/verify_saved_results.py --results results/my_run
python examples/budget_sensitivity.py --results results/my_run --output results/my_budget_study
python examples/quantile_comparison.py --results results/my_run --output results/my_quantiles
python -m unittest discover -s tests -v
```

Establezca también un hilo BLAS/OMP para los ejemplos si desea comparar
tiempos. Los tiempos dependen del hardware y la carga; las comparaciones
adicionales se ejecutaron con procesos concurrentes y sus tiempos no
constituyen un benchmark de rendimiento aislado.

## Archivos verificables

- `protocol.json`: parámetros y huellas del código antes de ejecutar el estudio.
- `system*/datasets.npz`: todas las trayectorias base nuevas y sus fases.
- `system*/model.npz`: modelos completos y calibraciones congeladas.
- `system*/optimizer_history.csv`: cada evaluación del objetivo, incluidas
  dos evaluaciones adicionales de auditoría (inicio/final), externas a `nfev`.
- `system*/updates_y*.json`: lotes, reglas, excitación, reducciones, fallback,
  cocientes y errores de Joseph.
- `trajectory_hashes.csv`: semillas y huellas de los datos de calibración y
  confirmación, reproducibles con `make_run`.
- `independent_verification.json`: verificaciones de inclusión en
  entrenamiento, cotas de complejidad y coherencia de resultados guardados.
- `executed_source.zip`: copia exacta del código existente al iniciar el
  experimento, con huellas coincidentes con el protocolo.

Después del experimento se añadieron comprobaciones de tipos de configuración,
rechazo de membresías fraccionarias y limpieza de un estado nominal previo
al reiniciar `fit_nominal`. No cambian las operaciones numéricas del protocolo
ejecutado. La distribución final añade además comparaciones y documentación.
Las predicciones de los modelos guardados se verificaron exactamente con
el código final. `release_manifest.json` identifica los archivos finales.

`tests/fixtures/matlab_reference.npz` es un oráculo de pruebas extraído de los
archivos del autor. Contiene valores históricos solo para contrastar la
conversión; ningún flujo de entrenamiento depende de él. Sus archivos de
procedencia y huellas están en `tests/fixtures/provenance.json`.

## Construir e instalar la distribución

```bash
python -m pip install build
python -m build
python -m pip install dist/manfiz_toolbox-0.1.0-py3-none-any.whl
```

El wheel contiene el código reutilizable. La distribución fuente incluye
documentación, ejemplos y pruebas; el ZIP de entrega añade los modelos,
datos y resultados para evitar imponer su tamaño a toda instalación.
Las instrucciones de empaquetado siguen la
[guía oficial de pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).

El árbol ya tiene configuración de paquete, README, CITATION, control de
archivos generados y workflow de pruebas. Puede cargarse como contenido de
un repositorio nuevo. La elección de licencia corresponde al autor y se
dejó identificada en `NOTICE.md`. No se creó repositorio remoto, DOI ni
publicación PyPI durante este trabajo.

## Interpretación para el manuscrito

Esta entrega aporta una implementación reproducible y entrenamiento nuevo.
No certifica aceptación Q1. Para incorporar sus números al paper deben
identificarse como un experimento nuevo, con PCG64, presupuesto actualizado,
dos terminaciones por límite de iteraciones y el alcance condicional de las
cotas. Las tablas del manuscrito anterior no se sobrescriben automáticamente.
