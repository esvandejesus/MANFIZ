# 0.1.0 - 2026-09-13

- Conversión nativa del flujo MATLAB a paquete instalable Python.
- Entrenamiento nuevo de las premisas compartidas y consecuentes nominales de
  los tres sistemas; reinicio y ajuste de todos los consecuentes zonotópicos.
- API para fit, predicción nominal/posterior, intervalos y calibración.
- Calibración por factibilidad conjunta con coeficientes de ambos signos,
  verificación primal/dual y comprobación del testigo en los zonotopos finales.
- Propagación explícita del ruido actual y de los retardos medidos.
- Reducción externa ponderada con selección por factibilidad y reporte de fallback.
- Datos NARX causales por experimento, normalización solo de entrenamiento y
  matrices de pesos calculadas de forma estable en logaritmos.
- Persistencia NPZ/JSON versionada; reentrenar invalida calibraciones previas.
- Rechazo de conteos fraccionarios, cotas incompatibles y dimensiones inválidas.
- Pruebas de geometría, causalidad, entrenamiento, ruido, serialización y
  referencia MATLAB; scripts de estudio q_max y cuantiles.
- Resultados nuevos de 540 trayectorias de confirmación, figuras PNG/SVG,
  documentación matemática y registro explícito de no convergencia en S1/S2.
