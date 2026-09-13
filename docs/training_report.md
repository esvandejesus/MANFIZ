# Informe de conversión y entrenamiento nuevo MANFIZ

Versión 0.1.0 - 13 de septiembre de 2026.

## Qué se hizo y por qué antes no se reentrenaron las premisas

La auditoría anterior conservó las premisas guardadas para comparar la reproducción con MATLAB. Sí repitió la actualización zonotópica con la cota corregida, pero no fue un entrenamiento completo desde una malla inicial. Esta entrega resuelve ese pendiente: los tres sistemas generaron datos nuevos y entrenaron premisas, consecuentes nominales y todos los zonotopos en Python.

Se entregan una API instalable, simuladores, calibración, predicción, serialización, ejemplos, 21 pruebas, datos, semillas, trazas, figuras y dos estudios adicionales. El flujo de entrenamiento no lee modelos MATLAB ni las referencias de prueba.

## Entrenamiento ejecutado

| Sistema | J inicial | J final | Mejora | Iteraciones | Evaluaciones | Terminación |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.000451607313 | 0.000425449319 | 5.79 % | 6000 | 7693 | Límite de iteraciones |
| 2 | 0.00139199106 | 0.00105174388 | 24.44 % | 6000 | 7699 | Límite de iteraciones |
| 3 | 0.000970105652 | 0.000907673394 | 6.44 % | 4286 | 5581 | Convergencia numérica |

Se usaron 7080 filas de entrenamiento por sistema, 1770 de validación y 1770 de evaluación retenida; 8 reglas, 18 parámetros de premisa y 11 coeficientes por regla/salida. La función objetivo disminuyó en los tres casos. S1 y S2 no cumplieron el criterio de convergencia antes del límite de 6000 iteraciones. No se afirma óptimo global.

## Cota total de regresión y consistencia conjunta

| Sistema | Salida | Mínimo LP físico dentro del prior | Cota total usada |
|---|---:|---:|---:|
| 1 | 1 | 0.033099649 | 0.034754631 |
| 1 | 2 | 0.028733335 | 0.030170002 |
| 2 | 1 | 0.026479487 | 0.027803461 |
| 2 | 2 | 0.025720491 | 0.027006516 |
| 3 | 1 | 0.028046276 | 0.029448589 |
| 3 | 2 | 0.028725387 | 0.030161656 |

La cota del sensor de entrenamiento es 0.017320508. Por sí sola resulta insuficiente en los seis canales dentro de sus priors calibrados. Se calcularon cotas totales compatibles con un único vector de correcciones para todas las filas, con margen del 5 %. Esto no prueba que esas cotas se cumplan fuera de entrenamiento.

Hubo 2406 actualizaciones locales: 810/816/780 en S1/S2/S3. Se verificó la pertenencia del testigo común a los 48 zonotopos finales. No hubo violaciones entre los 42480 valores de entrenamiento al usar radio paramétrico más cota total de entrenamiento.

## Cobertura independiente con ruido

Se calibró sobre 180 trayectorias limpias. Después de congelar los modelos y las cotas, se evaluaron 540 trayectorias separadas: **0 violaciones en 1,911,600 valores medidos**. Se usaron tres amplitudes y tres intensidades de ruido uniforme, con 20 réplicas de fase por combinación.

La holgura mínima medida fue 0.054525048. La envolvente limpia también pasó la confirmación sin violaciones, con holgura mínima 0.027893977.

La anchura media pasó de 0.386877 a 0.508028, un aumento del 31.32 %. La fórmula añade defecto limpio, ruido actual y efecto de los retardos al radio paramétrico.

Las muestras contabilizadas son k=30,...,1799, tras historia inicial/descarte. Se emparejan fases e intensidades de ruido; los valores temporales no se tratan como observaciones mutuamente independientes. La cobertura para todos los ruidos acotados sigue siendo condicional a una envolvente limpia válida. No es una garantía universal sobre cualquier trayectoria futura ni sobre ruido gaussiano sin recorte.

## Muestra previamente señalada

La medición 0.594191438453 del sistema 2, salida 1, muestra MATLAB 1770, queda en [0.334923473323, 0.716682827611] con el modelo nuevo. Su holgura es 0.122491389159. La trayectoria retrospectiva completa tuvo cero violaciones. Esta comprobación usa un caso conocido y se informa por separado de las 540 trayectorias de confirmación.

## Hallazgos y correcciones implementadas

| Hallazgo | Corrección/verificación |
|---|---|
| Faltaba ajuste nuevo de premisas | `fit()` inicializa y entrena el flujo completo |
| Cota del sensor confundida con error total | LP conjunto, cotas de ambos signos y verificación primal/dual |
| Inclusión puntual no implica parámetro común | Testigo común y pertenencia a los zonotopos finales |
| Ruido en los retardos ignorado al ampliar intervalos | Cota del cambio del centro y del radio paramétrico |
| Riesgo de mezclar normalización o retardos | Estadísticas solo de entrenamiento y NARX por experimento |
| Productos de membresías pueden subdesbordar | Cálculo en dominio logarítmico |
| Conteos fraccionarios podían truncarse o fallar tarde | Validación explícita de cantidades enteras |
| Reentrenar podía dejar calibraciones anteriores | Invalidación del estado dependiente del ajuste previo |
| Terminación del optimizador podía confundirse con convergencia | Reporte explícito de presupuesto/tolerancias |
| Reducción no siempre satisface ambos umbrales | Registro de cada fallback y verificación separada del límite de complejidad |
| Dependencia de objetos MATLAB | Modelos NPZ/JSON sin pickle y API independiente |

La reducción tuvo 653 fallbacks en 712 eventos. Conserva inclusión externa y límite de generadores; en esos eventos no se garantiza inflación <=1.10 y cociente posterior/propagado <=1.005 simultáneamente. El máximo posterior observado fue 1016 = q_max + 16. Los 59 eventos con candidato factible cumplieron el umbral de ciclo.

## Comparaciones y comprobaciones

Los presupuestos 800/1000/1200 generaron nueve ajustes de consecuentes. Comparten las premisas recién entrenadas para aislar el efecto de q_max; no son nueve entrenamientos adicionales de premisas. Todos los canales de las nueve variantes tuvieron PICP=1 en la evaluación retenida en modo validation. Más generadores redujeron las anchuras en estos casos; no se eligió el presupuesto a partir de la confirmación.

La comparación por cuantiles nominales del 95 % tuvo PICP entre 90.96 % y 92.88 % sobre la trayectoria de amplitud 1.15, con intervalos considerablemente más estrechos. No se presenta como una garantía determinista ni se afirma superioridad universal de un método.

El centro zonotópico no garantiza menor RMSE que MANFIS: por ejemplo, en S1/y1 pasó de 0.014204568 a 0.014352289 en la evaluación retenida. El objetivo de los conjuntos es expresar incertidumbre y consistencia, no asegurar una mejora puntual.

La equivalencia con las 18 simulaciones deterministas MATLAB tuvo diferencia máxima 4.45e-16; los centros de referencia se reprodujeron con diferencia máxima 1.34e-15. La persistencia de los modelos nuevos conservó exactamente centros y radios. La batería incluye 21 pruebas de propiedades matemáticas, causalidad, entrenamiento, ruido y referencias numéricas. La instalación se registra en `results/installation_verification.json`.

## Entrega y límites para publicación

El ZIP contiene fuentes, wheel, distribución fuente, ejemplos, pruebas, datos, modelos, trazas, figuras y documentación. No se publicaron repositorios ni paquetes. La licencia queda por definir por el autor. Las cifras de este experimento nuevo no sobrescriben automáticamente el manuscrito anterior. La toolbox no certifica aceptación Q1 ni elimina las hipótesis de las cotas.

```bash
python -m pip install -e ".[all]"
manfiz-benchmark --output results/my_run --max-iter 6000 --max-evaluations 10000
python -m unittest discover -s tests -v
```

Consulte `README.md`, `docs/api.md`, `docs/mathematics.md` y `docs/reproducibility.md` para el uso y las definiciones completas.
