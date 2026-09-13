"""Generate the Spanish delivery report from the actual exported metrics.

Optional authoring dependency: reportlab. It is not required by the toolbox.
Run from the project root after all experiments and checks.
"""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape
import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Preformatted


ROOT=Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT/name).read_text())


def main():
    system_fonts=Path('/usr/share/fonts/truetype/dejavu')
    bundled_fonts=Path(reportlab.__file__).parent/'fonts'
    for name,system,bundled in [('ReportSans','DejaVuSans.ttf','Vera.ttf'),
                                ('ReportSansBold','DejaVuSans-Bold.ttf','VeraBd.ttf'),
                                ('ReportMono','DejaVuSansMono.ttf','Vera.ttf')]:
        path=system_fonts/system
        if not path.exists():
            path=bundled_fonts/bundled
        pdfmetrics.registerFont(TTFont(name,str(path)))
    pdfmetrics.registerFontFamily('ReportSans',normal='ReportSans',bold='ReportSansBold',
                                  italic='ReportSans',boldItalic='ReportSansBold')
    training=load('results/fresh_training/training_report.json')
    summary=load('results/fresh_training/summary.json')
    historical=load('results/historical_sample_check.json')
    with (ROOT/'results/fresh_training/heldout_metrics.csv').open() as f:
        held=list(csv.DictReader(f))
    with (ROOT/'results/quantile_comparison/metrics.csv').open() as f:
        quantile=list(csv.DictReader(f))
    increase=100*(summary['mean_noise_mpiw']/summary['mean_validation_mpiw']-1)
    md=['# Informe de conversión y entrenamiento nuevo MANFIZ',
        '', 'Versión 0.1.0 - 13 de septiembre de 2026.', '',
        '## Qué se hizo y por qué antes no se reentrenaron las premisas', '',
        'La auditoría anterior conservó las premisas guardadas para comparar la reproducción con MATLAB. '
        'Sí repitió la actualización zonotópica con la cota corregida, pero no fue un entrenamiento completo '
        'desde una malla inicial. Esta entrega resuelve ese pendiente: los tres sistemas generaron datos '
        'nuevos y entrenaron premisas, consecuentes nominales y todos los zonotopos en Python.', '',
        'Se entregan una API instalable, simuladores, calibración, predicción, serialización, ejemplos, '
        '21 pruebas, datos, semillas, trazas, figuras y dos estudios adicionales. El flujo de entrenamiento '
        'no lee modelos MATLAB ni las referencias de prueba.', '',
        '## Entrenamiento ejecutado', '',
        '| Sistema | J inicial | J final | Mejora | Iteraciones | Evaluaciones | Terminación |',
        '|---|---:|---:|---:|---:|---:|---|']
    for t in training:
        gain=100*(1-t['final_objective']/t['initial_objective'])
        md.append(f"| {t['System']} | {t['initial_objective']:.9g} | {t['final_objective']:.9g} | {gain:.2f} % | {t['iterations']} | {t['function_evaluations']} | {'Convergencia numérica' if t['success'] else 'Límite de iteraciones'} |")
    md += ['', 'Se usaron 7080 filas de entrenamiento por sistema, 1770 de validación y 1770 de '
           'evaluación retenida; 8 reglas, 18 parámetros de premisa y 11 coeficientes por regla/salida. '
           'La función objetivo disminuyó en los tres casos. S1 y S2 no cumplieron el criterio de '
           'convergencia antes del límite de 6000 iteraciones. No se afirma óptimo global.', '',
           '## Cota total de regresión y consistencia conjunta', '',
           '| Sistema | Salida | Mínimo LP físico dentro del prior | Cota total usada |',
           '|---|---:|---:|---:|']
    for t in training:
        for o in range(2):
            md.append(f"| {t['System']} | {o+1} | {t['minimum_joint_error'][o]:.9f} | {t['regression_bound'][o]:.9f} |")
    md += ['', 'La cota del sensor de entrenamiento es 0.017320508. Por sí sola resulta insuficiente '
           'en los seis canales dentro de sus priors calibrados. Se calcularon cotas totales compatibles '
           'con un único vector de correcciones para todas las filas, con margen del 5 %. Esto no '
           'prueba que esas cotas se cumplan fuera de entrenamiento.', '',
           'Hubo 2406 actualizaciones locales: 810/816/780 en S1/S2/S3. Se verificó la pertenencia '
           'del testigo común a los 48 zonotopos finales. No hubo violaciones entre los 42480 valores '
           'de entrenamiento al usar radio paramétrico más cota total de entrenamiento.', '',
           '## Cobertura independiente con ruido', '',
           'Se calibró sobre 180 trayectorias limpias. Después de congelar los modelos y las cotas, '
           f"se evaluaron 540 trayectorias separadas: **0 violaciones en {summary['measured_channel_values']:,} valores medidos**. "
           'Se usaron tres amplitudes y tres intensidades de ruido uniforme, con 20 réplicas de fase por combinación.', '',
           f"La holgura mínima medida fue {summary['minimum_noisy_slack']:.9f}. La envolvente limpia también pasó la confirmación "
           f"sin violaciones, con holgura mínima {summary['minimum_clean_slack']:.9f}.", '',
           f"La anchura media pasó de {summary['mean_validation_mpiw']:.6f} a {summary['mean_noise_mpiw']:.6f}, un aumento del {increase:.2f} %. "
           'La fórmula añade defecto limpio, ruido actual y efecto de los retardos al radio paramétrico.', '',
           'Las muestras contabilizadas son k=30,...,1799, tras historia inicial/descarte. Se emparejan '
           'fases e intensidades de ruido; los valores temporales no se tratan como observaciones '
           'mutuamente independientes. La cobertura para todos los ruidos acotados sigue siendo '
           'condicional a una envolvente limpia válida. No es una garantía universal sobre cualquier '
           'trayectoria futura ni sobre ruido gaussiano sin recorte.', '',
           '## Muestra previamente señalada', '',
           f"La medición {historical['Measurement']:.12f} del sistema 2, salida 1, muestra MATLAB 1770, "
           f"queda en [{historical['Lower']:.12f}, {historical['Upper']:.12f}] con el modelo nuevo. "
           f"Su holgura es {historical['Slack']:.12f}. La trayectoria retrospectiva completa tuvo cero "
           'violaciones. Esta comprobación usa un caso conocido y se informa por separado de las '
           '540 trayectorias de confirmación.', '',
           '## Hallazgos y correcciones implementadas', '',
           '| Hallazgo | Corrección/verificación |', '|---|---|',
           '| Faltaba ajuste nuevo de premisas | `fit()` inicializa y entrena el flujo completo |',
           '| Cota del sensor confundida con error total | LP conjunto, cotas de ambos signos y verificación primal/dual |',
           '| Inclusión puntual no implica parámetro común | Testigo común y pertenencia a los zonotopos finales |',
           '| Ruido en los retardos ignorado al ampliar intervalos | Cota del cambio del centro y del radio paramétrico |',
           '| Riesgo de mezclar normalización o retardos | Estadísticas solo de entrenamiento y NARX por experimento |',
           '| Productos de membresías pueden subdesbordar | Cálculo en dominio logarítmico |',
           '| Conteos fraccionarios podían truncarse o fallar tarde | Validación explícita de cantidades enteras |',
           '| Reentrenar podía dejar calibraciones anteriores | Invalidación del estado dependiente del ajuste previo |',
           '| Terminación del optimizador podía confundirse con convergencia | Reporte explícito de presupuesto/tolerancias |',
           '| Reducción no siempre satisface ambos umbrales | Registro de cada fallback y verificación separada del límite de complejidad |',
           '| Dependencia de objetos MATLAB | Modelos NPZ/JSON sin pickle y API independiente |', '',
           'La reducción tuvo 653 fallbacks en 712 eventos. Conserva inclusión externa y límite de '
           'generadores; en esos eventos no se garantiza inflación <=1.10 y cociente posterior/propagado '
           '<=1.005 simultáneamente. El máximo posterior observado fue 1016 = q_max + 16. '
           'Los 59 eventos con candidato factible cumplieron el umbral de ciclo.', '',
           '## Comparaciones y comprobaciones', '',
           'Los presupuestos 800/1000/1200 generaron nueve ajustes de consecuentes. Comparten las '
           'premisas recién entrenadas para aislar el efecto de q_max; no son nueve entrenamientos '
           'adicionales de premisas. Todos los canales de las nueve variantes tuvieron PICP=1 en '
           'la evaluación retenida en modo validation. Más generadores redujeron las anchuras '
           'en estos casos; no se eligió el presupuesto a partir de la confirmación.', '',
           f"La comparación por cuantiles nominales del 95 % tuvo PICP entre {100*min(float(r['PICP']) for r in quantile):.2f} % "
           f"y {100*max(float(r['PICP']) for r in quantile):.2f} % sobre la trayectoria de amplitud 1.15, "
           'con intervalos considerablemente más estrechos. No se presenta como una garantía '
           'determinista ni se afirma superioridad universal de un método.', '',
           'El centro zonotópico no garantiza menor RMSE que MANFIS: por ejemplo, en S1/y1 pasó '
           'de 0.014204568 a 0.014352289 en la evaluación retenida. El objetivo de los conjuntos '
           'es expresar incertidumbre y consistencia, no asegurar una mejora puntual.', '',
           'La equivalencia con las 18 simulaciones deterministas MATLAB tuvo diferencia máxima '
           '4.45e-16; los centros de referencia se reprodujeron con diferencia máxima 1.34e-15. '
           'La persistencia de los modelos nuevos conservó exactamente centros y radios. '
           'La batería incluye 21 pruebas de propiedades matemáticas, causalidad, entrenamiento, '
           'ruido y referencias numéricas. La instalación se registra en '
           '`results/installation_verification.json`.', '',
           '## Entrega y límites para publicación', '',
           'El ZIP contiene fuentes, wheel, distribución fuente, ejemplos, pruebas, datos, '
           'modelos, trazas, figuras y documentación. No se publicaron repositorios ni paquetes. '
           'La licencia queda por definir por el autor. Las cifras de este experimento nuevo '
           'no sobrescriben automáticamente el manuscrito anterior. La toolbox no certifica '
           'aceptación Q1 ni elimina las hipótesis de las cotas.', '',
           '```bash', 'python -m pip install -e ".[all]"',
           'manfiz-benchmark --output results/my_run --max-iter 6000 --max-evaluations 10000',
           'python -m unittest discover -s tests -v', '```', '',
           'Consulte `README.md`, `docs/api.md`, `docs/mathematics.md` y '
           '`docs/reproducibility.md` para el uso y las definiciones completas.', '']
    markdown='\n'.join(md)
    (ROOT/'docs/training_report.md').write_text(markdown,encoding='utf-8')

    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyES',fontName='ReportSans',fontSize=9.8,leading=13.4,
                               spaceAfter=8,textColor=colors.HexColor('#243444')))
    styles.add(ParagraphStyle(name='SmallES',parent=styles['BodyES'],fontSize=8.6,leading=11,spaceAfter=5))
    styles.add(ParagraphStyle(name='TitleES',fontName='ReportSansBold',fontSize=23,leading=28,
                               textColor=colors.HexColor('#173e5a'),spaceAfter=12))
    styles.add(ParagraphStyle(name='HeadingES',fontName='ReportSansBold',fontSize=12.4,leading=17,
                               textColor=colors.HexColor('#173e5a'),spaceBefore=10,spaceAfter=8))
    story=[]
    def p(text,style='BodyES'):
        story.append(Paragraph(text,styles[style]))
    def h(text): p(text,'HeadingES')
    def table(rows,widths):
        values=[[Paragraph(escape(str(c)),styles['SmallES']) for c in row] for row in rows]
        t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf3')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                              ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),
                              ('BOTTOMPADDING',(0,0),(-1,-1),5),
                              ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#9bb2c3')),
                              ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d9e1e6'))]))
        story.append(t); story.append(Spacer(1,8))
    p('MANFIZ en Python','TitleES')
    p('Conversión, entrenamiento nuevo y verificación de intervalos<br/>Versión 0.1.0 | 13 de septiembre de 2026','SmallES')
    p('<b>Se entrenaron desde cero los tres sistemas</b>: premisas compartidas, consecuentes nominales y zonotopos de las dos salidas. La toolbox es instalable y funciona sin MATLAB ni archivos .mat.')
    p('La auditoría anterior conservó las premisas para contrastar la reproducción con los resultados guardados y repitió la actualización zonotópica. Eso dejó pendiente el ajuste completo desde la malla inicial, que sí se ejecutó en esta entrega.')
    h('Entrenamiento real, con terminación explícita')
    tab=[['Sist.','Iteraciones','Evaluaciones','Mejora de J','Terminación']]
    for t in training:
        tab.append([t['System'],t['iterations'],t['function_evaluations'],f"{100*(1-t['final_objective']/t['initial_objective']):.2f} %",'Convergencia numérica' if t['success'] else 'Límite de iteraciones'])
    table(tab,[48,70,79,88,226])
    p('Se usaron 7080 filas de entrenamiento por sistema, 1770 de validación y 1770 retenidas para evaluación; 8 reglas, 18 parámetros de premisa y 11 coeficientes por regla/salida. S1 y S2 alcanzaron el límite de 6000 iteraciones. No se afirma optimalidad global.')
    h('La cota del sensor no basta para el ajuste conjunto')
    tab=[['Sistema / salida','Mínimo LP físico','Cota total utilizada']]
    for t in training:
        for o in range(2):
            tab.append([f"S{t['System']} / y{o+1}",f"{t['minimum_joint_error'][o]:.9f}",f"{t['regression_bound'][o]:.9f}"])
    table(tab,[135,188,188])
    p('La cota del sensor de entrenamiento es 0.017320508. En los seis canales, el mínimo error conjunto dentro del prior es mayor. Se fijó una cota total compatible con un parámetro común y margen del 5 %. Esto establece consistencia en entrenamiento, no una cota universal fuera de él.','SmallES')
    story.append(PageBreak())
    p('Cobertura medida y caso señalado','TitleES')
    h('0 violaciones entre 1.911.600 valores con ruido')
    p('Después de calibrar con 180 trayectorias limpias, se congelaron las cotas y se evaluaron 540 trayectorias separadas. Se usaron amplitudes 1.00/1.15/1.30 y desviaciones de ruido uniforme 0.005/0.010/0.020. Cada serie aporta 1770 instantes con predicción, después de los primeros 30 de historia/descarte.')
    table([['Comprobación','Resultado'],
           ['Violaciones medidas / confirmación','0 / 1.911.600 valores'],
           ['Holgura mínima medida',f"{summary['minimum_noisy_slack']:.9f}"],
           ['Violaciones de la envolvente limpia','0'],
           ['Anchura media sin/con extensión',f"{summary['mean_validation_mpiw']:.6f} / {summary['mean_noise_mpiw']:.6f}"],
           ['Aumento de anchura media',f'{increase:.2f} %']], [277,234])
    p('La extensión suma al radio paramétrico el defecto limpio, el ruido actual y el efecto del ruido en los retardos. <b>Su garantía para todos los ruidos acotados es condicional a una envolvente limpia válida.</b> El estudio finito no prueba cobertura de cualquier trayectoria futura. Las fases y los patrones de ruido están emparejados entre condiciones.')
    h('La medición 0.594191 queda dentro del intervalo nuevo')
    p(f"Sistema 2, salida 1, muestra MATLAB 1770: intervalo <b>[{historical['Lower']:.6f}; {historical['Upper']:.6f}]</b>, con holgura {historical['Slack']:.6f}. Toda esa trayectoria tuvo cero violaciones. Es una comprobación retrospectiva de un caso conocido, separada de las 540 series de confirmación.")
    story.append(Image(str(ROOT/'results/historical_sample_check.png'),width=511,height=187.1))
    story.append(PageBreak())
    p('Verificaciones y entrega','TitleES')
    h('Correcciones incorporadas')
    table([['Problema o riesgo','Corrección/verificación'],
           ['Faltaba entrenar premisas nuevas','Inicialización y ajuste completo; parámetros iniciales/finales y trazas guardados'],
           ['Cota puntual confundida con factibilidad común','LP con coeficientes de ambos signos; revisión primal/dual y testigo común'],
           ['Ruido en retardos medidos','Se acota el cambio del centro y del radio paramétrico'],
           ['Dependencia de MATLAB y estado previo','Paquete NumPy/SciPy; modelos NPZ/JSON; invalidación de calibraciones al reajustar'],
           ['Reducción presentada siempre como factible','Fallback explícito; cotas de complejidad verificadas por separado'],
           ['Reproducibilidad y equivalencia','21 pruebas; datos, semillas, huellas, figuras y comparación con referencias MATLAB']], [179,332])
    p('Se verificaron 2406 actualizaciones, la pertenencia del testigo común a 48 zonotopos finales y cero violaciones sobre 42480 valores de entrenamiento usando su cota total. El máximo posterior fue 1016 generadores. Hubo 653 fallbacks en 712 reducciones: conservan inclusión externa y complejidad, pero no garantizan ambos umbrales de inflación/contracción.')
    p('Las simulaciones deterministas de referencia coinciden hasta 4.45e-16 y los centros guardados de MATLAB hasta 1.34e-15. Guardar y cargar los modelos nuevos conserva exactamente centros y radios. Se incluyen nueve ajustes de presupuestos 800/1000/1200 con premisas comunes nuevas, una comparación por cuantiles y simulaciones libres de centros.')
    h('Instalación y uso')
    story.append(Preformatted('python -m pip install -e ".[all]"\nmanfiz-benchmark --output results/my_run \\\n  --max-iter 6000 --max-evaluations 10000\npython -m unittest discover -s tests -v',
                               ParagraphStyle(name='CodeES',fontName='ReportMono',fontSize=8.4,leading=12,
                                              backColor=colors.HexColor('#f0f4f7'),borderPadding=8)))
    story.append(Spacer(1,10))
    p('El ZIP contiene código fuente, wheel, distribución fuente, ejemplos, pruebas, documentación y todos los resultados nuevos. README.md explica la API; docs/mathematics.md define las variables y demuestra la implicación de propagación del ruido; docs/training_report.md aporta el detalle completo. La comprobación de instalación está en results/installation_verification.json.','SmallES')
    p('El repositorio no se ha publicado. La licencia queda por definir por el autor. Esta entrega no certifica aceptación Q1: deben declararse las dos paradas por presupuesto, los fallbacks y el alcance condicional de las cotas. Las cifras nuevas no sobrescriben automáticamente el manuscrito anterior.','SmallES')

    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#cdd9e1')); canvas.line(42,37,A4[0]-42,37)
        canvas.setFont('ReportSans',8); canvas.setFillColor(colors.HexColor('#566f81'))
        canvas.drawString(42,24,'MANFIZ Toolbox 0.1.0 | Informe de verificación')
        canvas.drawRightString(A4[0]-42,24,str(doc.page))
    output=ROOT/'Informe_Toolbox_Python_MANFIZ.pdf'
    doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=49,
                         title='MANFIZ: conversión Python y entrenamiento nuevo',author='Informe preparado para Esvan-Jesús Pérez-Pérez')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(output)


if __name__=='__main__':
    main()
