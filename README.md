# Detección de anomalías en un banco hidráulico · Trabajo final INF-8239

**Tema 39 · VAE para detección de anomalías** · Alexis De La Cruz · Matrícula 100805370
Maestría en Ciencias de Datos e Inteligencia Artificial · UASD

> ¿Puede un detector de anomalías avisar de un componente que se degrada **antes** de que
> una aguja del panel entre en zona roja, y hace falta un VAE para conseguirlo?

## Las dos respuestas

**Sí avisa antes.** Donde el panel de agujas ve 8 de cada 100 válvulas con retardo pequeño,
el detector las ve todas.

**No hace falta un VAE.** La distancia de Mahalanobis gana **27 de 30** comparaciones
repetidas con diez particiones sobre tres objetivos. Lo que decidió el resultado no fue el
modelo sino **la representación**.

| Válvula · 435 ciclos de prueba | PR-AUC | ROC-AUC | Recall | Falsas alarmas |
|---|---|---|---|---|
| Alarma por aguja *(baseline)* | 0.8010 | 0.5512 | 0.142 | 1.8 % |
| **Mahalanobis** | **0.9974** | **0.9934** | **1.000** | 2.7 % |
| VAE | 0.9966 | 0.9912 | 0.870 | 2.7 % |

Línea de azar del PR-AUC: **0.745**. Sin declararla, un 0.80 parece bueno y no lo es.

## ¿Es real la diferencia, o fue suerte?

Una sola partición no distingue «Mahalanobis gana» de «Mahalanobis ganó esta vez». El
experimento se repitió con **diez particiones** sobre los **tres objetivos**, cambiando
solo qué ciclos caen en entrenamiento y cuáles en prueba.

| Objetivo | Mahalanobis gana | VAE gana | Alarma por aguja |
|---|---|---|---|
| Válvula | 7 de 10 | 3 de 10 | 0 |
| Fuga de bomba | **10 de 10** | 0 | 0 |
| Acumulador | **10 de 10** | 0 | 0 |

**La parte incómoda:** en la válvula, los intervalos de confianza del PR-AUC se solapan —
Mahalanobis [0.9870, 0.9961] frente a VAE [0.9863, 0.9918]—, de modo que **por PR-AUC la
diferencia no es distinguible del ruido**. Medido una sola vez parecía una victoria; medido
diez veces, no lo es.

**Lo que sí distingue, y con holgura, es el recall y la estabilidad:**

| Válvula | Recall medio | IC 95 % | Detección del retardo pequeño |
|---|---|---|---|
| **Mahalanobis** | **0.9997** | [0.9991, 1.0003] | **0.999 ± 0.003** |
| VAE | 0.8768 | [0.8049, 0.9488] | 0.631 ± **0.348** |
| Alarma por aguja | 0.1170 | [0.1034, 0.1305] | — |

Esa desviación de **0.348** es el hallazgo: el VAE no detecta peor, **detecta de forma
inestable**. En unas particiones ve el retardo pequeño y en otras casi no. Mahalanobis
tiene una desviación de 0.003.

El argumento correcto no es «Mahalanobis tiene mejor PR-AUC» —que en la válvula no se
sostiene— sino **«Mahalanobis es estable y el VAE no»**, que se sostiene en los tres
objetivos.

Evidencia en `reports/robustez/`, generada por `scripts/run_seeds.py`.

## El hallazgo: la representación pesó más que el modelo

La válvula es un componente de **conmutación**. Un retardo cambia **cuándo** sube la presión
dentro del ciclo de 60 segundos, no su promedio. Resumir el ciclo con media, mínimo y máximo
borra exactamente esa información:

| Representación | ROC-AUC |
|---|---|
| Media del ciclo por sensor | 0.58 |
| **Seis tramos temporales** | **0.996** |

Mismo modelo, mismos datos, mismo protocolo. De casi azar a casi perfecto.

## Un caso: lo que el operador no habría visto

Ciclo **1408**, válvula con retardo pequeño. El detector lo marca con 13.8 frente a un
umbral de 9.3. **Ninguna de las 17 agujas está en zona roja**: la que más se acerca,
, se queda al **36 %** de su límite.

Lo que lo delata son las presiones y la potencia de refrigeración **en el primer sexto del
ciclo**, que es exactamente donde se manifiesta un retardo de conmutación. Ninguna de esas
lecturas, por separado, sale de lo normal. Es la combinación la que no existe en una
máquina sana.

Y no es un caso elegido con pinzas: **260 de los 324 ciclos degradados** del conjunto de
prueba se detectan sin que ninguna aguja salga de su banda. El **80 %** de lo que el
sistema encuentra es invisible para el panel.

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
├── tests/                        56 pruebas
└── reports/                      métricas y evidencia de cada ejecución
```

## Pruebas

```bash
uv run pytest -q
```

56 pruebas: contrato de datos, prevención de fuga, extracción temporal, baseline,
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
