# Detección de anomalías en un banco hidráulico · Trabajo final INF-8239

**Tema 39 · VAE para detección de anomalías** · Alexis De La Cruz · Matrícula 100805370
Maestría en Ciencias de Datos e Inteligencia Artificial · UASD

> ¿Puede un detector de anomalías avisar de un componente que se degrada **antes** de que
> una aguja del panel entre en zona roja, y hace falta un VAE para conseguirlo?

## Las dos respuestas

**Sí avisa antes.** Donde el panel de agujas ve 8 de cada 100 válvulas con retardo pequeño,
el detector las ve todas.

**No hace falta un VAE.** La distancia de Mahalanobis le gana en los dos objetivos y en las
dos ablaciones. Lo que decidió el resultado no fue el modelo sino **la representación**.

| Válvula · 435 ciclos de prueba | PR-AUC | ROC-AUC | Recall | Falsas alarmas |
|---|---|---|---|---|
| Alarma por aguja *(baseline)* | 0.8010 | 0.5512 | 0.142 | 1.8 % |
| **Mahalanobis** | **0.9974** | **0.9934** | **1.000** | 2.7 % |
| VAE | 0.9966 | 0.9912 | 0.870 | 2.7 % |

Línea de azar del PR-AUC: **0.745**. Sin declararla, un 0.80 parece bueno y no lo es.

## El hallazgo: la representación pesó más que el modelo

La válvula es un componente de **conmutación**. Un retardo cambia **cuándo** sube la presión
dentro del ciclo de 60 segundos, no su promedio. Resumir el ciclo con media, mínimo y máximo
borra exactamente esa información:

| Representación | ROC-AUC |
|---|---|
| Media del ciclo por sensor | 0.58 |
| **Seis tramos temporales** | **0.996** |

Mismo modelo, mismos datos, mismo protocolo. De casi azar a casi perfecto.

## Detección temprana

| Estado de la válvula | Panel de agujas | Mahalanobis |
|---|---|---|
| Retardo pequeño | 8.3 % | **100 %** |
| Retardo severo | 9.3 % | 100 % |
| Cerca del fallo total | 25.0 % | 100 % |

El panel solo empieza a ver algo cuando la válvula ya está casi rota — que es cuando el
operador también lo vería.

## Datos

| | |
|---|---|
| Conjunto | Condition Monitoring of Hydraulic Systems · [UCI 447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) |
| Licencia | CC BY 4.0 · DOI 10.24432/C5CW21 · donado en 2018 por ZeMA gGmbH |
| Naturaleza | **Reales**, banco instrumentado con 17 sensores de presión, caudal, temperatura, potencia y vibración |
| Tamaño | 2205 ciclos de 60 s · 43 680 lecturas por ciclo |
| Trazabilidad | El script calcula el SHA-256 del zip y de los 18 archivos |

Los datos **no se versionan**: `scripts/download_data.py` los obtiene y verifica.

## Cuatro decisiones de preparación, tomadas antes de medir

1. **Objetivo: la válvula** (y el acumulador como caso difícil). Se descartó el
   refrigerador porque un solo sensor lo delata: CE correlaciona 0.992 y TS1 −0.911. Un
   operador lo vería en el termómetro, así que no sirve para la pregunta.
2. **Solo ciclos estables.** Los 1080 degradados son todos estables y de los 1125 óptimos
   solo 369 lo son: incluir los transitorios dejaría aprender «transitorio ⟹ sano» y
   acertar 756 casos sin entender nada.
3. **Partición estratificada, no cronológica.** El archivo está ordenado por estado del
   refrigerador (correlación del índice +0.884): un corte temporal entrenaría con un solo
   régimen. La partición correcta depende de la estructura del dato, no de una regla.
4. **CE, CP y SE marcados como virtuales.** No se miden, se calculan. Se conservan y se
   corre la ablación que comprueba que el resultado no depende de ellos: 0.9973 frente a
   0.9974.

## El límite que más importa

El detector del **acumulador solo funciona con la máquina ya caliente**:

| Estado del refrigerador | Recall |
|---|---|
| 100 % · eficiencia plena | 0.009 |
| 20 % · reducida | 0.019 |
| 3 % · cerca del fallo total | **0.585** |

Verificado que no es un artefacto del experimento: el acumulador está balanceado
120/120/120 entre los tres estados. El detector de la válvula **no** tiene este problema:
recall 1.000 en todos los regímenes.

## Ejecución

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/build_features.py
uv run pytest -q
```

```bash
uv run python scripts/run_models.py --objetivo valvula
uv run python scripts/run_models.py --objetivo acumulador
uv run python scripts/run_models.py --objetivo valvula --sin-virtuales
uv run python scripts/run_decision.py --objetivo valvula --razon-costo 10
uv run streamlit run app.py
```

## El DSS

`app.py` ordena la cola de inspección para un **analista de control** que decide a qué
equipos dedica la capacidad del turno. Muestra los umbrales y de dónde salen, explica cada
ciclo indicando **qué sensor y en qué tramo** se aparta, y declara sus límites en la propia
pantalla. Tiene tres zonas: inspeccionar, **vigilar** y normal; la zona de vigilancia es la
distancia entre lo que conviene revisar y lo que cabe en el turno, y un ciclo sin régimen
estacionario se abstiene siempre.

## Estructura

```
INF8239_TF/
├── app.py                        DSS en Streamlit
├── docs/MODEL_CARD.md            ficha del modelo elegido
├── scripts/
│   ├── download_data.py          descarga y huellas SHA-256
│   ├── build_features.py         de 43 680 lecturas a 112 variables por ciclo
│   ├── run_models.py             baseline, Mahalanobis y VAE con el mismo protocolo
│   └── run_decision.py           umbral por costo, abstención, régimen y Green AI
├── src/inf8239_tf/
│   ├── data.py                   contrato, extracción y partición
│   ├── baseline.py               la alarma por aguja de la planta
│   ├── modelos.py                Mahalanobis y VAE
│   ├── evaluacion.py             PR-AUC, umbral y severidad
│   └── decision.py               costo, capacidad, abstención y auditoría por régimen
├── tests/                        54 pruebas
└── reports/                      métricas y evidencia de cada ejecución
```

## Pruebas

```bash
uv run pytest -q
```

54 pruebas: contrato de datos, prevención de fuga, extracción temporal, baseline,
métricas, costo, abstención, auditoría por régimen, explicación por variable, arranque del
tablero sin navegador y **consistencia de la documentación** — estas últimas leen las
cifras de `reports/` y comprueban que son las que aparecen en este README y en el Model
Card, de modo que una ejecución nueva que cambie un resultado rompe la suite en lugar de
dejar un documento desactualizado.

La verificación de citas, licencias y reproducibilidad está en
[`docs/VERIFICACION.md`](docs/VERIFICACION.md).

## Entorno

Windows 11 · Intel Core i9-13980HX con 16 GB de RAM · Python 3.12 gestionado por uv ·
scikit-learn, PyTorch 2.14 en CPU y Streamlit.

## Declaración de uso de asistentes de IA

Este trabajo se desarrolló con la asistencia de Claude Opus 5 (Anthropic). El detalle está
en [`docs/USO_DE_IA.md`](docs/USO_DE_IA.md).

## Licencia y cita

Material académico para INF-8239. El conjunto se usa bajo CC BY 4.0, citando: Helwig, N.,
Pignanelli, E., & Schütze, A. (2015). *Condition monitoring of a complex hydraulic system
using multivariate statistics*. 2015 IEEE International Instrumentation and
Measurement Technology Conference (I2MTC) Proceedings, 210-215.
https://doi.org/10.1109/I2MTC.2015.7151267
