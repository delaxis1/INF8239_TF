"""DSS de priorización de anomalías para el analista de control.

El decisor es una persona que cada turno elige **a qué equipos dedica una capacidad de
inspección limitada**. El sistema ordena la cola y explica por qué, con sus umbrales a la
vista y una zona de abstención declarada. No autoriza ninguna acción automática.

Uso:
    uv run streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from inf8239_tf.data import (OBJETIVOS, RAIZ, columnas_aguja, columnas_sensor, etiquetar,
                             particion, solo_normales)
from inf8239_tf.baseline import AlarmaPorAguja
from inf8239_tf.decision import (Costos, Politica, auditoria_por_regimen,
                                 resumen_de_politica, umbral_por_capacidad,
                                 umbral_por_costo)
from inf8239_tf.evaluacion import linea_de_azar
from inf8239_tf.modelos import Mahalanobis

CICLOS = RAIZ / "data" / "ciclos.csv"
COLORES = {"inspeccionar": "#9B2C2C", "abstencion": "#B7791F", "normal": "#2F6F4E"}

st.set_page_config(page_title="DSS de priorización de anomalías", layout="wide")
st.title("Priorización de anomalías en un banco hidráulico")
st.warning(
    "**Apoya la decisión, no la sustituye.** El sistema ordena una cola de inspección a "
    "partir de datos **reales pero de un banco de pruebas de laboratorio**, no de una "
    "planta en producción. Ninguna salida autoriza intervenir, parar un equipo ni "
    "declarar un fallo: solo propone un orden para que una persona decida."
)


@st.cache_data(show_spinner=False)
def cargar() -> pd.DataFrame:
    return pd.read_csv(CICLOS)


@st.cache_resource(show_spinner="Ajustando el detector con ciclos normales…")
def preparar(objetivo: str, capacidad: float, razon: float):
    tabla = etiquetar(cargar(), objetivo)
    entrena, prueba = particion(tabla, objetivo=objetivo)
    normales = solo_normales(entrena)
    corte = int(len(normales) * 2 / 3)
    ajuste, validacion_normal = normales.iloc[:corte], normales.iloc[corte:]
    validacion = pd.concat([validacion_normal, entrena[entrena.anomalia == 1]])

    variables = columnas_sensor(tabla)
    maha = Mahalanobis(variables).ajustar(ajuste)
    agujas = AlarmaPorAguja.ajustar(ajuste, columnas_aguja(tabla))

    umbral_costo = umbral_por_costo(validacion.anomalia.to_numpy(),
                                    maha.puntuar(validacion), Costos(razon))
    puntuacion = maha.puntuar(prueba)
    politica = Politica.desde_costo_y_capacidad(
        umbral_costo, umbral_por_capacidad(puntuacion, capacidad))
    return tabla, prueba, maha, agujas, puntuacion, politica, umbral_costo, len(ajuste)


with st.sidebar:
    st.header("Configuración")
    disponibles = sorted(OBJETIVOS, key=lambda o: o != "valvula")
    objetivo = st.selectbox("Componente vigilado", disponibles,
                            format_func=lambda o: f"{o} · {OBJETIVOS[o]['descripcion']}")
    capacidad = st.slider("Capacidad de inspección del turno", 0.05, 0.50, 0.15, 0.05,
                          format="%.0f%%", help="Fracción de ciclos que el equipo puede "
                                                "revisar. Fija el umbral de inspección.")
    razon = st.select_slider("Un fallo no detectado cuesta…", [1, 2, 5, 10, 20, 50, 100],
                             value=10, format_func=lambda v: f"{v}× una inspección inútil")
    st.caption("Los dos valores son decisiones de la institución, no del modelo. Cambiarlos "
               "mueve el umbral y el sistema lo muestra.")

tabla, prueba, maha, agujas, puntuacion, politica, umbral_costo, n_ajuste = preparar(
    objetivo, capacidad, razon)
decision = politica.decidir(puntuacion, prueba.estable.to_numpy())
definicion = OBJETIVOS[objetivo]

st.subheader("Qué decide el sistema y con qué umbrales")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ciclos en la cola", len(prueba))
c2.metric("A inspeccionar", int((decision == "inspeccionar").sum()))
c3.metric("En vigilancia", int((decision == "abstencion").sum()))
c4.metric("Declarados normales", int((decision == "normal").sum()))
st.caption(
    f"Umbral de inspección **{politica.umbral_inspeccion:.2f}**, fijado por la capacidad "
    f"del {capacidad:.0%}. Umbral de costo **{umbral_costo:.2f}**, el que minimizaría el "
    f"costo con la razón {razon}:1. Entre ambos está la **zona de vigilancia**: ciclos que "
    f"convendría revisar y que no caben en el turno. El sistema no los declara normales."
)

st.subheader("Cola de inspección priorizada")
orden = np.argsort(-puntuacion)
cola = prueba.iloc[orden].copy()
cola["puntuacion"] = puntuacion[orden]
cola["decision"] = decision[orden]
cola["aguja_mas_desviada"] = agujas.aguja_culpable(prueba.iloc[orden])
visibles = ["ciclo", "puntuacion", "decision", "aguja_mas_desviada", "severidad"]
st.dataframe(
    cola[visibles].head(40).style.map(
        lambda v: f"color: {COLORES.get(v, '')}", subset=["decision"]),
    use_container_width=True, hide_index=True)
st.caption("`severidad` es la etiqueta real del banco de pruebas y se muestra solo para "
           "poder auditar el sistema. En operación no existiría: es justo lo que se quiere "
           "averiguar.")

st.subheader("Por qué este ciclo")
elegido = st.selectbox("Ciclo de la cola", cola.ciclo.head(40).tolist())
fila = prueba[prueba.ciclo == elegido]
aportes = maha.contribuciones(fila).iloc[0].sort_values(ascending=False)
izq, der = st.columns([1, 1])
with izq:
    estado = cola.loc[cola.ciclo == elegido, "decision"].iloc[0]
    st.markdown(f"**Decisión:** :{'red' if estado == 'inspeccionar' else 'orange' if estado == 'abstencion' else 'green'}[{estado}]")
    st.metric("Puntuación", f"{cola.loc[cola.ciclo == elegido, 'puntuacion'].iloc[0]:.2f}")
    st.write("**Variables que más explican la distancia:**")
    st.dataframe(aportes.head(8).rename("aporte").to_frame(), use_container_width=True)
with der:
    st.write("**Cómo leerlo.** El nombre indica sensor y tramo del ciclo: `PS2_t3` es la "
             "presión PS2 en el tercer sexto de los 60 segundos. Un aporte alto significa "
             "que ese sensor, **en ese momento del ciclo**, se aparta de lo que hace una "
             "máquina sana.")
    st.write("La aguja más desviada se muestra aparte porque es lo que el operador vería "
             "en el panel. Cuando el sistema marca un ciclo y ninguna aguja está en rojo, "
             "eso **es** el valor que aporta: ha visto una combinación imposible de "
             "lecturas normales.")

st.subheader("Qué pasa en cada nivel de degradación")
st.dataframe(resumen_de_politica(prueba, decision), use_container_width=True,
             hide_index=True)

st.subheader("Límites: lo que este sistema no permite concluir")
auditoria = auditoria_por_regimen(prueba, puntuacion, politica.umbral_inspeccion,
                                  "refrigerador")
peor, mejor = auditoria.recall.min(), auditoria.recall.max()
st.markdown(
    f"- **Depende del régimen de operación.** El recall de este detector va de "
    f"**{peor:.3f} a {mejor:.3f}** según el estado del refrigerador. Si la brecha es "
    f"grande, el sistema solo ve el fallo bajo ciertas condiciones de la máquina.\n"
    f"- **La prevalencia del banco no es la de una planta.** Aquí el "
    f"{linea_de_azar(prueba.anomalia.to_numpy()):.1%} de los ciclos está degradado porque "
    f"es un experimento diseñado. Con la prevalencia real de una planta, la precisión de "
    f"la cola sería mucho menor.\n"
    f"- **Un ciclo sin régimen estacionario nunca recibe veredicto.** El detector se "
    f"entrenó con {n_ajuste} ciclos estables; fuera de ese régimen se abstiene.\n"
    f"- **Marcar no es diagnosticar.** El sistema dice que un ciclo se aparta de lo "
    f"normal, no qué componente falla ni por qué.\n"
    f"- **Datos de laboratorio.** Nada de esto estima el comportamiento sobre una máquina "
    f"en producción.")
st.dataframe(auditoria, use_container_width=True, hide_index=True)

st.subheader("Acción humana")
st.info(
    "Revisar los ciclos de la cola en el orden propuesto, confirmar o descartar cada uno "
    "en el equipo físico y **registrar el resultado junto a la puntuación del sistema**. "
    "Los ciclos en vigilancia no se cierran: vuelven a la cola del turno siguiente. "
    "Si la tasa de confirmación cae por debajo de lo acordado, se retira el sistema y la "
    "priorización vuelve al orden por fecha."
)
