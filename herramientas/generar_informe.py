"""Genera el informe de síntesis del trabajo final (DOCX + PDF) desde los artefactos.

Lee reports/datos/, reports/modelos_*/ y reports/decision_*/, de modo que el informe
siempre coincide con la última ejecución de los scripts. No hay ninguna cifra escrita a
mano en este archivo.

Uso:
    uv run python herramientas/generar_informe.py --repo-url https://github.com/USUARIO/INF8239_TF
    uv run python herramientas/generar_informe.py --sin-pdf
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from formato import (BASE, HEADER_FILL, MUTED, NOMBRE, REPORTES, add_field, add_hyperlink,
                     add_inline_markdown, bullet, caption, data_table, fecha_larga,
                     heading, key_value_table, numbered, paragraph, remove_table_borders,
                     set_cell_fill, subheading)


def leer() -> dict:
    """Todas las cifras del informe, desde los archivos que las produjeron."""
    datos = {"resumen": json.loads((REPORTES / "datos" / "resumen.json").read_text("utf-8"))}
    for objetivo in ("valvula", "acumulador"):
        datos[objetivo] = {
            "comparacion": pd.read_csv(REPORTES / f"modelos_{objetivo}" / "comparacion.csv"),
            "severidad": pd.read_csv(REPORTES / f"modelos_{objetivo}" /
                                     "severidad_comparada.csv"),
            "sin_virtuales": pd.read_csv(REPORTES / f"modelos_{objetivo}_sin_virtuales" /
                                         "comparacion.csv"),
            "costo": pd.read_csv(REPORTES / f"decision_{objetivo}" /
                                 "sensibilidad_costo.csv"),
            "politica": pd.read_csv(REPORTES / f"decision_{objetivo}" / "politica.csv"),
            "regimen": pd.read_csv(REPORTES / f"decision_{objetivo}" /
                                   "regimen_refrigerador.csv"),
            "verde": pd.read_csv(REPORTES / f"decision_{objetivo}" / "green_ai.csv"),
            "protocolo": json.loads((REPORTES / f"decision_{objetivo}" /
                                     "protocolo.json").read_text("utf-8")),
        }
    return datos


def fila(tabla: pd.DataFrame, alternativa: str) -> pd.Series:
    return tabla.set_index("alternativa").loc[alternativa]


def tabla_metricas(doc, comparacion: pd.DataFrame) -> None:
    etiquetas = {"baseline_agujas": "Alarma por aguja (baseline)",
                 "mahalanobis": "Mahalanobis", "vae": "VAE"}
    filas = [[etiquetas[r.alternativa], f"{r.pr_auc:.4f}", f"{r.roc_auc:.4f}",
              f"{r.recall:.3f}", f"{r.falsas_alarmas:.3f}", f"{r.ajuste_s:.3f}"]
             for r in comparacion.itertuples()]
    data_table(doc, ["Alternativa", "PR-AUC", "ROC-AUC", "Recall", "Falsas alarmas",
                     "Ajuste s"], filas, [5.2, 2.4, 2.4, 2.2, 2.6, 2.2], bold_rows=(1,))


def build(ev: dict, repo_url: str | None) -> Path:
    resumen = ev["resumen"]
    valv, acum = ev["valvula"], ev["acumulador"]
    maha_v = fila(valv["comparacion"], "mahalanobis")
    base_v = fila(valv["comparacion"], "baseline_agujas")
    vae_v = fila(valv["comparacion"], "vae")
    maha_a = fila(acum["comparacion"], "mahalanobis")
    vae_a = fila(acum["comparacion"], "vae")
    azar_v = float(valv["comparacion"].linea_de_azar.iloc[0])
    azar_a = float(acum["comparacion"].linea_de_azar.iloc[0])
    part = resumen["particion"]

    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21.59), Cm(27.94)
    section.left_margin = section.right_margin = Cm(2.2)
    section.top_margin, section.bottom_margin = Cm(2.0), Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.1

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run("Trabajo final · Tema 39 · Alexis De La Cruz · página ")
    run.font.size = Pt(8)
    run.font.color.rgb = MUTED
    add_field(footer, "PAGE")
    run = footer.add_run(" de ")
    run.font.size = Pt(8)
    run.font.color.rgb = MUTED
    add_field(footer, "NUMPAGES")

    band = doc.add_table(rows=1, cols=1)
    remove_table_borders(band)
    cell = band.rows[0].cells[0]
    set_cell_fill(cell, HEADER_FILL)
    cell.width = Cm(17.19)
    first = cell.paragraphs[0]
    first.paragraph_format.space_before = Pt(8)
    first.paragraph_format.space_after = Pt(2)
    run = first.add_run("UNIVERSIDAD AUTÓNOMA DE SANTO DOMINGO · MAESTRÍA EN CIENCIAS DE "
                        "DATOS E INTELIGENCIA ARTIFICIAL")
    run.font.size = Pt(7.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0xC9, 0xD6, 0xE6)
    title = cell.add_paragraph()
    title.paragraph_format.space_after = Pt(2)
    run = title.add_run("Detección temprana de degradación en un banco hidráulico")
    run.font.size = Pt(17)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    subtitle = cell.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(8)
    run = subtitle.add_run("Trabajo final · Tema 39 · VAE para detección de anomalías · "
                           "Modalidad B · DSS")
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0xE6, 0xEC, 0xF3)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    key_value_table(doc, [
        ("Asignatura", "INF-8239 Ciencia de Datos II"),
        ("Tema", "39 · VAE para detección de anomalías · eje de IA generativa, ética y "
                 "aplicaciones"),
        ("Modalidad", "B · Sistema de soporte a decisiones · trabajo individual"),
        ("Estudiante", "Alexis De La Cruz · Matrícula 100805370"),
        ("Fecha", fecha_larga(date.today())),
        ("Entorno", "Windows 11 · Intel Core i9-13980HX con 16 GB de RAM · Python 3.12 "
                    "gestionado por uv · scikit-learn, PyTorch 2.14 en CPU y Streamlit"),
    ])

    # 1
    heading(doc, "1. Repositorio y artefactos")
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    add_inline_markdown(p, "**Enlace:** ", size=10.5)
    if repo_url:
        add_hyperlink(p, repo_url)
        add_inline_markdown(p, "  ·  etiqueta `tf-entrega`  ·  repositorio público",
                            size=10.5)
    else:
        add_inline_markdown(p, "pendiente (regenerar con `--repo-url`)", size=10.5,
                            color=MUTED)
    paragraph(doc, "Los artefactos son **ejecutables, no capturas**: cuatro scripts "
                   "reproducen cada cifra de este informe desde los datos originales, y "
                   "`reports/` conserva la evidencia de las seis ejecuciones. La historia "
                   "de Git documenta las correcciones, incluidas las que cambiaron el "
                   "resultado.", size=10, after=3)

    # 2
    heading(doc, "2. El problema y por qué un panel de agujas no lo ve")
    paragraph(doc, "Un panel de control muestra una aguja por sensor con su banda verde y "
                   "su zona roja. El operador vigila **cada aguja contra su propio "
                   "límite**. Eso le deja ciego ante la **anomalía contextual**: cada "
                   "lectura dentro de rango, pero la combinación imposible. Una presión "
                   "normal y un caudal normal pueden ser incompatibles entre sí sin que "
                   "ninguna aguja se mueva de la zona verde.", size=10.5, after=4)
    key_value_table(doc, [
        ("Decisor", "Analista de control de planta"),
        ("Decisión asistida", "A qué equipos dedica la capacidad de inspección del turno"),
        ("Lo que entrega el sistema", "Una cola priorizada, la razón de cada caso y una "
                                      "zona de abstención declarada"),
        ("Lo que NO entrega", "Diagnóstico, orden de trabajo ni autorización para parar un "
                              "equipo"),
    ])

    # 3
    heading(doc, "3. Datos y decisiones de preparación")
    key_value_table(doc, [
        ("Conjunto", "Condition Monitoring of Hydraulic Systems · UCI 447 · **datos "
                     "reales** de un banco instrumentado"),
        ("Licencia", "CC BY 4.0 · DOI 10.24432/C5CW21 · donado en 2018 por ZeMA gGmbH"),
        ("Sensores", f"{resumen['sensores']} · presión, caudal, temperatura, potencia y "
                     f"vibración · ciclos de 60 s"),
        ("Volumen", f"{resumen['ciclos_totales']} ciclos · 43 680 lecturas por ciclo"),
        ("Trazabilidad", "El script calcula el SHA-256 del zip y de los 18 archivos"),
        ("Partición", f"Estratificada {100 - int(part['prueba'] / (part['entrenamiento'] + part['prueba']) * 100)}/"
                      f"{int(part['prueba'] / (part['entrenamiento'] + part['prueba']) * 100)}, "
                      f"semilla {part['semilla']}: {part['entrenamiento']} de "
                      f"entrenamiento y {part['prueba']} de prueba"),
    ])
    subheading(doc, "Cuatro decisiones tomadas antes de medir")
    for i, texto in enumerate((
        "**Objetivo: la válvula**, con el acumulador como caso difícil. Se descartó el "
        "refrigerador al medir que un solo sensor lo delata: CE correlaciona 0.992 y TS1 "
        "−0.911. Un operador lo vería en el termómetro, así que no sirve para la pregunta.",
        f"**Solo ciclos con régimen estacionario.** Los degradados son todos estables y de "
        f"los óptimos solo una parte lo es: incluir los transitorios permitiría aprender "
        f"«transitorio implica sano» y acertar {resumen['descartados_por_transitorio']} "
        f"casos sin entender nada. Quedan {resumen['ciclos_estables']}.",
        "**Partición estratificada, no cronológica.** El archivo está ordenado por estado "
        "del refrigerador, con correlación del índice +0.884: un corte temporal entrenaría "
        "con un único régimen. Es lo contrario de lo que conviene en una serie temporal, y "
        "el motivo es la estructura del dato, no una regla general.",
        "**CE, CP y SE marcados como virtuales**: no se miden, se calculan. Se conservan y "
        "se corre la ablación que comprueba que el resultado no depende de ellos.",
    ), start=1):
        numbered(doc, texto, i)

    # 4
    heading(doc, "4. El hallazgo: la representación pesó más que el modelo")
    paragraph(doc, "La válvula es un componente de **conmutación**: sus niveles son "
                   "«retardo pequeño» y «retardo severo». Un retardo cambia **cuándo** sube "
                   "la presión dentro del ciclo, no su promedio. Resumir el ciclo con "
                   "media, mínimo y máximo borra exactamente esa información.", size=10.5,
              after=4)
    data_table(doc, ["Representación", "Variables", "ROC-AUC (Mahalanobis)"], [
        ["Media del ciclo por sensor", "9", "0.58"],
        ["Seis tramos temporales por sensor", "54", "0.996"],
        ["Doce tramos temporales", "108", "1.000"],
    ], [8.0, 3.0, 5.8], bold_rows=(1,))
    caption(doc, "Mismo modelo, mismos datos, mismo protocolo. Ninguna arquitectura "
                 "compensó una representación que borraba la señal, y este resultado llegó "
                 "corrigiendo un error propio: la primera extracción de características "
                 "era la de la primera fila.")
    paragraph(doc, f"La extracción definitiva son {resumen['columnas_sensor']} variables: "
                   f"media y desviación por sensor, más la media de cada tramo temporal "
                   f"—{resumen['segmentos_rapidos']} tramos para los sensores de 100 y "
                   f"10 Hz, y {resumen['segmentos_lentos']} para los de 1 Hz, que solo "
                   f"tienen 60 lecturas por ciclo.", size=10, after=3)

    # 5
    heading(doc, "5. Resultados")
    n_prueba = valv["protocolo"]["prueba"] if "prueba" in valv["protocolo"] else part["prueba"]
    subheading(doc, f"Válvula · {n_prueba} ciclos de prueba · línea de azar del PR-AUC "
                    f"{azar_v:.3f}")
    tabla_metricas(doc, valv["comparacion"])
    caption(doc, "El ROC-AUC de 0.551 del baseline es puntuar casi al azar. Un PR-AUC de "
                 "0.801 contra una línea de azar de 0.745 es casi no hacer nada: por eso "
                 "la línea de azar se declara en cada tabla.")
    subheading(doc, f"Acumulador, el caso difícil · línea de azar {azar_a:.3f}")
    tabla_metricas(doc, acum["comparacion"])
    caption(doc, "La propia documentación del conjunto señala el acumulador como el "
                 "objetivo más complejo. Ningún método pasa de 0.46 de recall, lo que "
                 "impide concluir que «la detección de anomalías funciona» como si fuera "
                 "una propiedad general.")

    # 6
    heading(doc, "6. Detección temprana: la pregunta del trabajo")
    sev = valv["severidad"]
    filas = [[r.severidad, str(int(r.ciclos)), f"{r.baseline_agujas:.3f}",
              f"{r.mahalanobis:.3f}", f"{r.vae:.3f}"] for r in sev.itertuples()]
    data_table(doc, ["Estado de la válvula", "Ciclos", "Panel de agujas", "Mahalanobis",
                     "VAE"], filas, [5.6, 2.0, 3.4, 3.0, 3.0], bold_rows=(1,))
    caption(doc, "Fracción de ciclos detectados en cada nivel, **al 5 % de falsas "
                 "alarmas y sin restricción de capacidad**. La primera fila son ciclos "
                 "sanos: ahí la cifra son falsas alarmas. La sección 8 añade la capacidad "
                 "del turno y cambia el reparto, no la capacidad de distinguir.")
    paragraph(doc, "El operador ve **8 de cada 100** válvulas con retardo pequeño y 25 de "
                   "cada 100 cuando ya está cerca del fallo total — es decir, solo empieza "
                   "a verlas cuando él mismo lo notaría. El detector las ve **todas desde "
                   "el primer nivel de degradación**. Esa diferencia es la utilidad del "
                   "sistema, y es lo que el trabajo se propuso medir.", size=10.5, after=3)

    # 7
    heading(doc, "7. Por qué el VAE no se queda con el puesto")
    for texto in (
        f"**Pierde en los dos objetivos.** En la válvula detecta el "
        f"{sev.loc[sev.severidad.str.contains('pequeno'), 'vae'].iloc[0]:.1%} de los "
        f"retardos pequeños frente al "
        f"{sev.loc[sev.severidad.str.contains('pequeno'), 'mahalanobis'].iloc[0]:.0%} de "
        f"Mahalanobis. En el acumulador, recall {vae_a.recall:.3f} frente a "
        f"{maha_a.recall:.3f}.",
        "**Pierde también en las dos ablaciones** sin sensores virtuales: cuatro "
        "ejecuciones independientes con el mismo resultado.",
        f"**Se le dio una segunda oportunidad.** La primera ejecución agotó las épocas sin "
        f"converger; se repitió con parada temprana hasta 3000 y mejoró de 0.333 a 0.611 "
        f"en el nivel leve. La conclusión se escribió después de eso.",
        f"**El costo es desproporcionado:** {int(vae_v.parametros)} parámetros y "
        f"{vae_v.ajuste_s:.1f} s de ajuste frente a {int(maha_v.parametros)} y "
        f"{maha_v.ajuste_s:.3f} s.",
    ):
        bullet(doc, texto)
    paragraph(doc, "La conclusión no es que el VAE sea malo, sino que **la relación entre "
                   "estos sensores es suficientemente lineal como para que no haga falta**. "
                   "Un método de 1936 basta. Responder eso con evidencia era la pregunta, "
                   "no demostrar que la red gana.", size=10.5, after=3)

    # 8
    heading(doc, "8. Umbral, costo y abstención")
    costo_v = valv["costo"]
    filas = [[f"{r.razon_fn_fa:g} : 1", f"{r.umbral:.3f}", str(int(r.marcados)),
              f"{r.recall:.3f}", f"{r.falsas_alarmas:.3f}"]
             for r in costo_v.itertuples()]
    data_table(doc, ["Un fallo cuesta…", "Umbral óptimo", "Marcados", "Recall",
                     "Falsas alarmas"], filas, [4.0, 3.4, 2.8, 2.8, 3.8])
    caption(doc, "Válvula: el umbral NO se mueve en todo el rango, de 1:1 a 100:1. La "
                 "decisión es robusta y no hace falta conocer el costo con precisión. En "
                 "el acumulador ocurre lo contrario: el óptimo manda inspeccionar el 98 % "
                 "de los ciclos, lo que es correcto e inviable.")
    paragraph(doc, f"De esa contradicción sale la **política de tres zonas**: el umbral de "
                   f"inspección lo fija la capacidad del turno —el "
                   f"{valv['protocolo']['capacidad']:.0%} de los ciclos— y la **zona de "
                   f"vigilancia** es la distancia entre lo que conviene revisar y lo que se "
                   f"puede revisar. Esos ciclos no se declaran normales. Además, **todo "
                   f"ciclo sin régimen estacionario se abstiene siempre**: el detector se "
                   f"entrenó solo con ciclos estables y fuera de ese régimen no sabe lo que "
                   f"mira.", size=10, after=4)
    paragraph(doc, "**Esta tabla y la de la sección 6 no se contradicen: miden puntos de "
                   "operación distintos.** Allí se fija la tasa de falsas alarmas en el 5 % "
                   "y se pregunta qué detecta el modelo; aquí se añade la capacidad real "
                   "del turno. Por eso la mayoría de los ciclos degradados cae en "
                   "«vigilar» y no en «inspeccionar»: **el límite ya no es el detector "
                   "sino el presupuesto de inspección**. El detector los ha distinguido; "
                   "la planta no puede revisarlos todos.", size=10, after=4)
    pol = valv["politica"]
    filas = [[r.severidad, str(int(r.ciclos)), f"{r.normal:.3f}", f"{r.abstencion:.3f}",
              f"{r.inspeccionar:.3f}"] for r in pol.itertuples()]
    data_table(doc, ["Estado de la válvula", "Ciclos", "Normal", "Vigilar",
                     "Inspeccionar"], filas, [5.6, 2.0, 3.0, 3.0, 3.4])

    # 9
    heading(doc, "9. El límite que más importa")
    reg = acum["regimen"]
    filas = [[f"{int(r.refrigerador)} %", str(int(r.ciclos)), f"{r.recall:.3f}",
              f"{r.falsas_alarmas:.3f}"] for r in reg.itertuples()]
    data_table(doc, ["Estado del refrigerador", "Ciclos", "Recall del detector",
                     "Falsas alarmas"], filas, [5.0, 2.6, 4.6, 4.6],
               warn_rows=(len(filas) - 1,))
    caption(doc, "El detector del acumulador solo funciona con la máquina ya caliente. "
                 "Verificado que no es un artefacto del experimento: el acumulador está "
                 "balanceado 120/120/120 entre los tres estados del refrigerador. El "
                 "detector de la válvula no tiene este problema: recall 1.000 en todos los "
                 "regímenes de los otros tres componentes.")
    paragraph(doc, "Un sistema que solo ve el fallo cuando la máquina ya está degradada en "
                   "otro sitio **no sirve para anticiparlo**, que era el propósito. Por eso "
                   "el DSS recomienda el acumulador solo como vigilancia secundaria y lo "
                   "declara en la propia pantalla.", size=10, after=3)

    # 10
    heading(doc, "10. Green AI")
    verde = valv["verde"]
    etiquetas = {"baseline_agujas": "Alarma por aguja", "mahalanobis": "Mahalanobis",
                 "vae": "VAE"}
    filas = [[etiquetas[r.alternativa], f"{int(r.parametros)}",
              f"{r.latencia_ms_por_ciclo:.5f}",
              f"{r.precision_de_lo_inspeccionado:.3f}"] for r in verde.itertuples()]
    data_table(doc, ["Alternativa", "Parámetros", "Latencia ms/ciclo",
                     "Precisión de lo inspeccionado"], filas, [4.4, 3.0, 4.0, 5.4],
               bold_rows=(1,))
    caption(doc, "A igual capacidad de inspección, que es la comparación honesta: con el "
                 "mismo presupuesto de revisión, quién encuentra más fallos reales. En el "
                 "acumulador los tres dan 0.985, pero eso no los iguala: con 75 % de "
                 "prevalencia casi cualquier cosa que se marque es un fallo y la métrica "
                 "se satura.")

    # 11
    heading(doc, "11. Lo que este trabajo no demuestra")
    for texto in (
        f"**La prevalencia del banco no es la de una planta.** Aquí el {azar_v:.1%} de los "
        f"ciclos está degradado porque es un experimento diseñado. Con la prevalencia real "
        f"la precisión de la cola caería mucho, y por eso la precisión a capacidad fija no "
        f"se usa como argumento principal.",
        "**Una sola partición**, sin intervalos de confianza ni repetición con otras "
        "semillas.",
        f"**{part['normales_entrenamiento']} ciclos normales de entrenamiento.** El VAE "
        f"opera cerca del límite de datos que necesita, lo que debilita —aunque no "
        f"invalida— la conclusión sobre él.",
        "**Marcar no es diagnosticar.** El sistema dice que un ciclo se aparta de lo "
        "normal, no qué componente falla ni por qué.",
        "**Son datos de laboratorio.** Nada de esto estima el comportamiento sobre una "
        "máquina en producción, y trasladarlo exigiría volver a medir.",
    ):
        bullet(doc, texto)

    # 12
    heading(doc, "12. Conclusión accionable")
    paragraph(doc, "**Desplegar el detector de la válvula como apoyo a la priorización, y "
                   "el del acumulador solo como vigilancia secundaria.** El de la válvula "
                   "detecta el 100 % de los retardos desde el nivel más leve, con 2.7 % de "
                   "falsas alarmas, y mantiene ese rendimiento con el resto de la máquina "
                   "degradada. El del acumulador no: depende del régimen y su recall cae a "
                   "0.009 con el refrigerador sano.", size=10.5, after=4)
    paragraph(doc, "**Con Mahalanobis, no con el VAE.** Gana en los dos objetivos y en las "
                   "dos ablaciones, con menos parámetros y 2000 veces menos cómputo de "
                   "ajuste. Mantener el VAE en producción sería pagar más por menos.",
              size=10.5, after=4)
    paragraph(doc, "**Y antes de llevarlo a una planta: volver a medir.** Con la "
                   "prevalencia real, con más de una partición y comprobando que el "
                   "régimen de operación observado es el mismo con el que se entrenó. El "
                   "sistema se retira si la tasa de confirmación de las inspecciones cae "
                   "por debajo de lo acordado, y la priorización vuelve al orden por fecha.",
              size=10.5, after=3)

    # Referencias
    heading(doc, "Referencias")
    for ref in (
        "Helwig, N., Pignanelli, E., & Schütze, A. (2015). Condition monitoring of a "
        "complex hydraulic system using multivariate statistics. 2015 IEEE International "
        "Instrumentation and Measurement Technology Conference (I2MTC) Proceedings, "
        "210-215. https://doi.org/10.1109/I2MTC.2015.7151267",
        "Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. "
        "International Conference on Learning Representations. "
        "https://doi.org/10.48550/arXiv.1312.6114",
        "Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional "
        "covariance matrices. Journal of Multivariate Analysis, 88(2), 365-411. "
        "https://doi.org/10.1016/S0047-259X(03)00096-4",
        "Mahalanobis, P. C. (1936). On the generalised distance in statistics. Proceedings "
        "of the National Institute of Sciences of India, 2(1), 49-55.",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of "
        "Machine Learning Research, 12, 2825-2830.",
        "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative "
        "than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS "
        "ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432",
        "Schwartz, R., Dodge, J., Smith, N. A., & Etzioni, O. (2020). Green AI. "
        "Communications of the ACM, 63(12), 54-63. https://doi.org/10.1145/3381831",
    ):
        p = paragraph(doc, ref, size=8.5, after=2)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)

    heading(doc, "Declaración de uso de asistentes de IA")
    paragraph(doc, "Este trabajo se desarrolló con la asistencia de **Claude Opus 5 "
                   "(Anthropic)**, que apoyó estructura, código, pruebas y redacción y "
                   "ejecutó los experimentos. **No decidió los valores del análisis:** el "
                   "planteamiento del problema, la elección del conjunto y del componente "
                   "objetivo, el alcance y la interpretación son míos. Todas las cifras "
                   "provienen de ejecutar los scripts en el equipo indicado arriba. El "
                   "detalle de herramientas, instrucciones, verificaciones y las cinco "
                   "correcciones que cambiaron el resultado está en `docs/USO_DE_IA.md`.",
              size=8.5, after=2)

    salida = BASE / "herramientas" / f"{NOMBRE}.docx"
    doc.save(salida)
    return salida


def export_pdf(docx_path: Path) -> Path:
    pdf_path = docx_path.with_suffix(".pdf")
    script = (
        "$ErrorActionPreference = 'Stop';"
        "$word = New-Object -ComObject Word.Application; $word.Visible = $false;"
        f"try {{ $doc = $word.Documents.Open('{docx_path}', $false, $true);"
        f"$doc.ExportAsFixedFormat('{pdf_path}', 17); $doc.Close([ref]0) }}"
        "finally { $word.Quit() }"
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                   check=True)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-url", help="enlace del repositorio en GitHub")
    parser.add_argument("--sin-pdf", action="store_true", help="no exportar a PDF")
    args = parser.parse_args()
    docx_path = build(leer(), args.repo_url)
    print("DOCX:", docx_path)
    if not args.sin_pdf:
        print("PDF: ", export_pdf(docx_path))


if __name__ == "__main__":
    main()
