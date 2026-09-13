# Correspondencia funcional del código suministrado

La conversión es funcional, con módulos reutilizables y una API Python. Los
nombres históricos ANFIZ/MANFIZ de los puntos de entrada se unifican en
`MANFIZ`; no se requieren envoltorios que llamen a MATLAB.

| Archivo MATLAB | Implementación Python |
|---|---|
| `main_manfiz_nonlinear_benchmarks.m` | `manfiz.cli.run_benchmarks` y `manfiz-benchmark` |
| `main_anfiz_nonlinear_benchmarks.m` | Mismo punto de entrada; nombre histórico unificado |
| `config_nonlinear_benchmarks.m` | `config.py`, argumentos de CLI y `protocol.json` |
| `simulate_nonlinear_benchmark_dataset.m` | `benchmarks.simulate`, `make_run`, `make_training_runs`, `multisine` |
| `build_nonlinear_regression.m` | `NARXSpec.transform` y `transform_runs` |
| `build_mimo_normalization.m` | `Standardizer.fit` para X e y |
| `normalize_X.m`, `normalize_Y.m` | `Standardizer.transform` |
| `denormalize_Y.m` | `Standardizer.inverse` |
| `train_shared_premise_fis.m` | `premises.train_premises` y `BellPremises` |
| `fit_shared_consequents_wls.m` | `premises.ridge_fit` |
| `anfiz_rule_weights.m` | `BellPremises.weights` |
| `eval_center_model.m` | `premises.evaluate_center` y `MANFIZ.predict` |
| `train_anfiz_batched_output.m` | `recursion.train_output` |
| `select_informative_batch.m` | Asignación, QR con pivoteo y comprobación PE en `train_output` |
| `zkf_batched_parameter_combastel.m` | `zonotopes.measurement_update` y `gain_and_covariation` |
| `zonotope_reduce_combastel.m` | `zonotopes.reduce_fixed` |
| `zonotope_reduce_feasibility_aware_combastel.m` | `zonotopes.reduce_adaptive` |
| `weighted_f_radius.m` | `zonotopes.weighted_radius` |
| `evaluate_anfiz_partition.m` | `MANFIZ.predict_interval` |
| `point_metrics_mimo.m` | `regression_metrics` |
| `export_benchmark_tables.m` | CSV/JSON de `cli.py` y `tools/summarize_results.py` |
| `make_benchmark_plots.m` | `plotting.py`; salidas PNG y SVG editables |
| `main_qmax_sensitivity_final.m` | `examples/budget_sensitivity.py` |
| `main_qmax_sensitivity_v2.m` | Mismo ejemplo; configuración final consolidada |
| `audit_common_parameter_bound.m` | `calibration.minimax_correction`, con o sin prior |
| `calibrate_manfiz_training_bounds.m` | `joint_training_bounds` y verificación de testigo |
| `simulate_manfiz_noise_case.m` | `benchmarks.make_run` y construcción NARX |
| `calibrate_manfiz_noise_envelope.m` | `MANFIZ.calibrate_noise` |
| `evaluate_manfiz_noise_envelope.m` | `MANFIZ.predict_interval(mode="noise")` |
| `run_noise_coverage_verification.m` | Calibración y confirmación separadas en `run_benchmarks` |
| `run_punto6_audit.m` | `tools/verify_saved_results.py` y las pruebas de referencia MATLAB |

Las comparaciones por cuantiles y la simulación libre, incorporadas durante la
revisión anterior, están en `baselines.py`, `NARXSpec.free_run` y los ejemplos.

## Diferencias intencionales y trazabilidad

1. Los sistemas y retardos se conservaron. La simulación determinista completa
   se contrastó con 18 trayectorias MATLAB: diferencia máxima 4.45e-16.
2. El ruido nuevo usa `default_rng` de NumPy con PCG64. No se intenta imitar
   exactamente el generador Twister de MATLAB; los datos y semillas nuevos
   se guardan para reproducción. Por ello los resultados nuevos no deben
   sustituirse sin explicación por las cifras del experimento archivado.
3. Los pesos se calculan en logaritmos. Para los parámetros archivados, las
   predicciones de prueba coinciden con los centros almacenados de MATLAB
   con diferencia máxima 1.34e-15. Ese contraste usa una referencia pequeña
   exclusivamente en `tests/fixtures`; `fit()` no la lee.
4. Nelder-Mead de SciPy y `fminsearch` no tienen que recorrer el mismo simplex
   ni terminar con idénticos parámetros. Se mantienen el tipo de optimizador
   y el objetivo; se registran inicialización, presupuesto y terminación.
5. La versión Python deriva una cota de error total compatible con una
   corrección común en entrenamiento. Distingue esa cota de la del sensor.
6. La extensión de ruido incluye tanto el sensor actual como los retardos de
   salida medidos y el cambio de radio paramétrico debido a esos retardos.
7. Se evitan archivos FIS/MCOS y pickle. Los modelos usan NPZ numérico y JSON
   versionado. La validación de dimensiones se realiza al cargar.

Los resultados archivados no se sobrescribieron. El paquete nuevo contiene
los resultados de entrenamiento nativo Python y una referencia numérica
reducida para pruebas; no contiene archivos MATLAB ejecutables ni `.mat`.
