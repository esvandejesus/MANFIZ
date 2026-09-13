# Formulación implementada y alcance de las cotas

## Notación y dimensiones

| Símbolo | Dimensión | Definición |
|---|---|---|
| N, d, m, r | escalares enteros | Número de filas, regresores, salidas y reglas |
| k, i, o, j | índices | Muestra, regla, salida y coordenada de regresor |
| x_k, y_k | d, m | Regresores y salidas en unidades físicas |
| mu_X, sigma_X | d | Media y desviación muestral de entrenamiento |
| mu_Y, sigma_Y | m | Media y desviación muestral de las salidas de entrenamiento |
| phi_k | n=d+1 | Vector `[(x_k-mu_X)/sigma_X, 1]` |
| a, b, c | escalares | Ancho positivo, exponente positivo y centro de una membresía campana |
| lambda_ki | escalar | Activación normalizada no negativa de la regla i; su suma es 1 |
| beta0_oi | n | Consecuente nominal normalizado de regla i y salida o |
| delta_oi | n | Corrección desconocida del consecuente nominal |
| c_oi, R_oi | n, n por p_i | Centro y generadores del conjunto de correcciones |
| xi_oi | p_i | Coordenadas del zonotopo, con norma infinito <= 1 |
| H | N por rn | Matriz de bloques `lambda_ki * phi_k.T` |
| e_o | N | Residuo nominal normalizado de la salida o |
| b_y, b_x | m, d | Amplitudes máximas del ruido actual y de la incertidumbre de regresores |
| b_train | m | Cota física del error total de regresión sobre entrenamiento |
| D_o | escalar | Cota física del defecto de la envolvente de salida limpia |
| W | n por n | Matriz simétrica definida positiva para el radio F ponderado |
| C, z, f | B por n, B, B | Matriz, observación corregida y cota total de un lote local |
| B, q_target, q_max | enteros | Filas del lote y presupuestos de generadores |

Las divisiones entre vectores de normalización son por coordenada. `R R.T`
es una matriz de covariación geométrica de generadores; no es una covarianza
probabilística. Todas las variables de coeficientes se expresan en el espacio
normalizado; los intervalos devueltos se desnormalizan por salida.

## Premisas y ajuste nominal

Cada membresía generalizada campana es

\[
\mu(x;a,b,c)=\frac{1}{1+|(x-c)/a|^{2b}},\qquad a,b>0.
\]

El peso de una regla es el producto de sus membresías, dividido por la suma
de productos de todas las reglas. Se calcula en el dominio logarítmico para
evitar subdesbordamiento de los productos. Las reglas constituyen el producto
cartesiano de las membresías, con el último índice variando más rápido.

La predicción nominal normalizada de la salida o es

\[
\widehat y^0_{k,o}=\sum_{i=1}^{r}\lambda_{ki}\phi_k^T\beta^0_{o,i}.
\]

Se inicializan las campanas con los extremos de entrenamiento: centros
equiespaciados, ancho igual a la mitad del espaciado y exponente b=2. Se optimiza
el vector p formado por `(log(a),log(b),c)` de cada membresía. Al evaluarlo se
limita a a en [0.001,50] y b en [0.1,20]. Para cada p se reajustan **todas** las
salidas mediante ridge con rho=1e-8:

\[
\beta^0(p)=(H(p)^T H(p)+\rho I)^{-1}H(p)^T Y_n.
\]

La implementación usa factorizaciones/soluciones lineales, no forma la inversa.
La búsqueda externa minimiza el error cuadrático medio conjunto normalizado
más una penalización suave. La penalización es 1e-4 multiplicado por la suma
de las medias de `[|log(a)|-4]_+^2`, `[|log(b)|-3]_+^2` y `[|c|-4]_+^2`.
La ridge determina los coeficientes internos; el objetivo externo usa el error
de predicción, sin añadir de nuevo la norma ridge de beta.

Se conserva Nelder-Mead, como en el código MATLAB. Sus presupuestos y estado de
terminación se guardan por separado del resultado numérico. Detenerse por
`maxiter` o `maxfev` no equivale a satisfacer las tolerancias. La convergencia
numérica tampoco prueba optimalidad global. Véase la
[documentación oficial de SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html).

## Prior puntual y factibilidad de un parámetro común

El prior de cada corrección es una caja centrada en cero. Su forma proviene
del máximo cambio de coeficientes entre los reajustes por experimento y el
ajuste nominal conjunto. Por regla/salida, se aplica un piso relativo del 2 %
del mayor cambio y se normaliza la forma en norma euclídea. Una forma
numéricamente nula se sustituye por el vector uniforme de norma uno.

Para una forma d_oi no negativa, se calcula

\[
g_{k,o}=\sum_i |\lambda_{ki}|\,|\phi_k|^T d_{o,i},\qquad
\alpha_o=\max_k\frac{[|e_{k,o}|-b_{y,o}/\sigma_{Y,o}]_+}{g_{k,o}}.
\]

Los radios iniciales son el máximo por coordenada entre
`1.05 * alpha_o * d_oi` y 1e-7. Un denominador nulo con necesidad de radio
positiva se rechaza. Esta construcción comprueba inclusión **punto a punto**,
pero cada punto podría estar usando un parámetro diferente.

Por ello se resuelve un problema adicional con una sola corrección apilada
delta_o para todas las filas:

\[
t_o^*=\min_{\delta_o\in\mathbb R^{rn},\ t\ge0}t,
\qquad -t\mathbf1\le e_o-H\delta_o\le t\mathbf1,
\qquad -r_o^0\le\delta_o\le r_o^0.
\]

Los coeficientes delta tienen cotas de ambos signos. La cota física de
entrenamiento se fija como el máximo entre la cota del sensor,
`1.05 * sigma_Y,o * t_o*` y el piso `1e-10 * sigma_Y,o`. Se revisan residuos
primales y brecha dual del LP. Las cotas de signo se declaran explícitamente:
la configuración predeterminada de `linprog` sería no negativa. Véase
[SciPy HiGHS](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html).

La solución delta_o sirve como **testigo común de entrenamiento**. No es una
prueba de que el mismo error total esté acotado así fuera de entrenamiento.
Reentrenar las premisas no elimina automáticamente esta distinción. En los
seis canales recién entrenados, la cota del sensor sola sigue siendo
insuficiente dentro de los priors calibrados.

## Actualización local de zonotopos

Cada muestra se asigna a la regla de mayor peso si este es al menos 0.20.
Para aceptar un lote, C debe tener rango n y cociente de valores singulares
menor/mayor de al menos 1e-3. Se usa selección QR con pivoteo cuando hay más
de B filas disponibles. Las filas usadas se eliminan; las rechazadas se
mantienen hasta exceder el tamaño de búfer. Estos búferes pueden acumular
filas de varios experimentos ya construidos de manera causal.

Para regla i y salida o, en cada fila del lote:

\[
C_k=\lambda_{ki}\phi_k^T,\quad
z_k=e_{k,o}-\sum_{j\ne i}\lambda_{kj}\phi_k^T c_{o,j},
\]

\[
f_k=b_{\mathrm{train},o}/\sigma_{Y,o}
    +\sum_{j\ne i}|\lambda_{kj}|\,\|\phi_k^T R_{o,j}\|_1.
\]

La segunda suma incluye explícitamente la incertidumbre de las reglas que
se solapan. Antes de la actualización se añade la deriva local configurada,
se reduce si corresponde y se calcula, con P=R R.T y F=diag(f):

\[
K=PC^T(CPC^T+FF^T)^{-1},\qquad A=I-KC,
\]

\[
c^+=c+K(z-Cc),\qquad R^+=[AR,\,-KF].
\]

Se verifica independientemente la identidad de Joseph
`R+ R+.T = A P A.T + K F F.T K.T`. Para cualquier punto del prior que
satisfaga la franja `|z-C delta| <= f`, la identidad anterior representa
delta dentro del nuevo zonotopo con coordenadas acotadas. La reducción
externa conserva esta inclusión. Con deriva cero se comprueba por LP la
pertenencia del testigo común a cada zonotopo final.

## Reducción externa y límite de complejidad

El radio usado para seleccionar reducciones es

\[
\rho_W(R)=\sqrt{\operatorname{tr}(W R R^T)}.
\]

Se ordenan generadores por `r_j.T W r_j`. Para presupuesto q se conservan
q-n generadores y se reemplazan los restantes por una caja diagonal cuyos
radios son las sumas por fila de sus valores absolutos. Esto es una inclusión
externa exacta en sentido conjuntista; el radio F puede aumentar.

Cuando el número propagado supera `q_trigger`, se examinan los presupuestos
desde `q_target` hasta `min(q_max,p-1)`. Se escoge el menor que satisface a la
vez inflación <=1.10 y cociente de radio posterior/propagado <=1.005, con
las tolerancias configuradas. Si ninguno sirve, se minimiza la suma de
cuadrados de las violaciones relativas positivas; los empates favorecen q
menor. Este caso se registra como `fallback=True`.

En un fallback sigue habiendo inclusión externa y límite de complejidad,
pero no se afirma cumplimiento de ambos umbrales. El límite posterior es
`q_max+B`, porque la actualización añade B generadores después de reducir.
En el estudio principal es 1016, no 1000. Los conteos q son números de
generadores, no órdenes del zonotopo divididos entre la dimensión.

## Intervalo paramétrico y extensión de ruido medido

En unidades físicas, el centro y radio paramétrico son

\[
c_o(x)=\mu_{Y,o}+\sigma_{Y,o}\sum_i\lambda_i(x)\phi(x)^T
                  (\beta^0_{o,i}+c_{o,i}),
\]

\[
r_{p,o}(x)=\sigma_{Y,o}\sum_i|\lambda_i(x)|\,
                            \|\phi(x)^T R_{o,i}\|_1.
\]

El modo `validation` añade una caja física igual a 1.05 veces el máximo
residuo absoluto del centro posterior en validación. Se conserva esta
construcción conservadora del flujo suministrado.

Sea x* el regresor limpio, x=x*+v_x el medido y y=y*+v_y. Se supone
`|v_x,j| <= b_x,j`, `|v_y,o| <= b_y,o` y que las premisas dependen solo de
coordenadas con b_x,j=0. Así, los pesos lambda son iguales en x* y x.
Además se necesita la envolvente limpia

\[
|y_o^*-c_o(x^*)|\le r_{p,o}(x^*)+D_o.
\]

Definiendo theta_oi=beta0_oi+c_oi, para las primeras d coordenadas (sin
intercepto), una cota suficiente de propagación por los regresores es

\[
h_o(x)=\sigma_{Y,o}\sum_{j=1}^{d}\frac{b_{x,j}}{\sigma_{X,j}}
\left(\left|\sum_i\lambda_i\theta_{o,i,j}\right|
      +\sum_i|\lambda_i|\,\|R_{o,i}[j,:]\|_1\right).
\]

El primer sumando acota el cambio del centro. El segundo acota la diferencia
entre el radio paramétrico limpio y el calculado con regresores medidos.
Por desigualdad triangular se obtiene

\[
|y_o-c_o(x)|\le r_{p,o}(x)+D_o+b_{y,o}+h_o(x).
\]

Esta es la fórmula de `mode="noise"`; no basta con añadir ruido al instante
actual e ignorar los retardos medidos. La suma cubre todos los valores dentro
de las cajas de ruido **si se cumple la hipótesis limpia**. No necesita
independencia probabilística del ruido para esa implicación determinista.

`calibrate_noise` fija D como 1.10 veces el máximo exceso positivo limpio
sobre el radio paramétrico, conservando como piso la caja de validación
salvo que se especifique otro. Este máximo es una calibración finita. La
validez limpia se vuelve a comprobar en trayectorias de confirmación separadas,
sin ampliar D a partir de sus resultados. No se afirma una prueba uniforme
de la envolvente limpia para todo el dominio de excitación.

## Interpretación experimental

PICP es la fracción de valores contenidos, MPIW es la anchura media y NMPIW
es MPIW dividido por el rango observado de la misma salida/serie. RMSE, MAE
y R2 se calculan por salida. R2 y NMPIW son indefinidos para determinados
objetivos constantes y se representan como NaN en memoria/null en JSON.
La tolerancia de conteo de violaciones es 1e-10 unidades físicas.

Las 540 series de confirmación reutilizan fases entre amplitudes y el patrón
de ruido entre intensidades. Están separadas de calibración, pero no son 540
réplicas mutuamente independientes. No se calcula una confianza binomial
tratando los millones de valores temporales como observaciones independientes.
Las cotas de un paso no se presentan como cotas de simulación libre.
