# Model Card · Detector de anomalías del banco hidráulico

**INF-8239 Ciencia de Datos II · Trabajo final · Tema 39** · Alexis De La Cruz · 100805370

| | |
|---|---|
| Modelo elegido | **Distancia de Mahalanobis** con encogimiento de Ledoit-Wolf |
| Candidato descartado | VAE de 7736 parámetros, latente 4 |
| Entrada | 112 variables por ciclo: media y desviación de 17 sensores, más la media de cada tramo temporal |
| Entrenamiento | **Solo ciclos normales**: 172 del componente vigilado |
| Fecha | 26 de octubre de 2026 |
| Artefactos | `reports/modelos_*`, `reports/decision_*` |

## Uso previsto

**Ordenar una cola de inspección** para que un analista de control decida a qué equipos
dedica una capacidad de revisión limitada en el turno. El sistema propone un orden y
explica cada caso. La decisión de inspeccionar, intervenir o archivar la toma una persona.

## Usos fuera de alcance

- **Diagnosticar.** El sistema dice que un ciclo se aparta de lo normal; **no dice qué
  componente falla ni por qué**.
- **Disparar acciones automáticas**: parar un equipo, emitir una orden de trabajo o
  declarar un fallo sin revisión humana.
- **Operar sobre una máquina en producción.** Los datos son de un banco de pruebas de
  laboratorio. Nada de lo medido aquí anticipa el comportamiento en planta.
- **Vigilar el acumulador bajo refrigeración normal**, por la razón de la sección de
  límites.
- **Evaluar ciclos que no alcanzaron régimen estacionario.** El sistema se abstiene.

## Datos y partición

| | |
|---|---|
| Fuente | Condition Monitoring of Hydraulic Systems · [UCI 447](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) · CC BY 4.0 · DOI 10.24432/C5CW21 |
| Naturaleza | **Reales**, de un banco instrumentado con 17 sensores, ciclos de 60 s |
| Ciclos | 2205 en origen · **1449 tras descartar los transitorios** |
| Partición | Estratificada 70/30 por nivel de degradación, semilla 42 |
| Trazabilidad | `scripts/download_data.py` guarda el SHA-256 del zip y de los 18 archivos |

Tres decisiones de preparación, todas tomadas antes de medir:

1. **Solo ciclos estables.** Los degradados son todos estables y de los óptimos solo lo es
   una parte: incluir los transitorios dejaría aprender «transitorio ⟹ sano» y acertar 756
   casos sin entender nada.
2. **Partición estratificada, no cronológica.** El archivo está ordenado por estado del
   refrigerador (correlación del índice +0.884), de modo que un corte temporal entrenaría
   con un único régimen. Es lo contrario de lo que conviene en una serie temporal, y el
   motivo es la estructura del dato.
3. **El perfil temporal, no solo los agregados.** Ver la sección siguiente.

## El hallazgo principal: la representación pesó más que el modelo

La válvula es un componente de **conmutación**: un retardo cambia **cuándo** sube la
presión dentro del ciclo, no su promedio.

| Representación | ROC-AUC (Mahalanobis) |
|---|---|
| Media del ciclo por sensor | 0.58 |
| **Seis tramos temporales** | **0.996** |

Pasar de una a otra mueve la detección de «casi azar» a «casi perfecta» **sin cambiar de
modelo**. Ninguna arquitectura compensó una representación que borraba la señal.

## Evaluación

Conjunto de prueba: 435 ciclos. Umbral fijado sobre validación al 5 % de falsas alarmas.
**Línea de azar del PR-AUC: la prevalencia**, 0.745 y 0.752.

### Válvula

| | PR-AUC | ROC-AUC | Recall | Falsas alarmas |
|---|---|---|---|---|
| Alarma por aguja *(baseline)* | 0.8010 | 0.5512 | 0.142 | 1.8 % |
| **Mahalanobis** | **0.9974** | **0.9934** | **1.000** | 2.7 % |
| VAE | 0.9966 | 0.9912 | 0.870 | 2.7 % |

### Acumulador

| | PR-AUC | ROC-AUC | Recall | Falsas alarmas |
|---|---|---|---|---|
| Alarma por aguja *(baseline)* | 0.8504 | 0.6160 | 0.352 | 3.7 % |
| **Mahalanobis** | **0.9574** | **0.9115** | **0.456** | 3.7 % |
| VAE | 0.9121 | 0.7829 | 0.284 | 3.7 % |

### Detección temprana, que es la pregunta del trabajo

| Válvula | Panel de agujas | Mahalanobis | VAE |
|---|---|---|---|
| Retardo pequeño | 8.3 % | **100 %** | 61.1 % |
| Retardo severo | 9.3 % | 100 % | 100 % |
| Cerca del fallo total | 25.0 % | 100 % | 100 % |

El operador ve 8 de cada 100 válvulas con retardo pequeño y 25 de cada 100 cuando ya está
casi rota. El modelo las ve todas desde el primer nivel.

## Por qué se descarta el VAE

Pierde en los dos objetivos y en las dos ablaciones sin sensores virtuales: cuatro
ejecuciones independientes. Se le dio convergencia con parada temprana hasta 3000 épocas,
la misma representación, el mismo umbral y el mismo protocolo.

La relación entre estos sensores resulta suficientemente **lineal** como para que un método
de 1936 con 6440 parámetros y 4 milésimas de segundo de ajuste supere a una red de 7736
parámetros que tarda 9.5 segundos. No es un fracaso del VAE: es la respuesta a la pregunta
de si hacía falta.

## Green AI

| | Parámetros | Ajuste | Latencia por ciclo |
|---|---|---|---|
| Alarma por aguja | 51 | 0.001 s | 0.004 ms |
| **Mahalanobis** | 6 440 | **0.004 s** | 0.006 ms |
| VAE | 7 736 | 9.5 s | 0.005 ms |

A igual capacidad de inspección (15 % de los ciclos), en la válvula Mahalanobis y el VAE
alcanzan precisión 1.000 frente a 0.879 del panel. En el acumulador los tres dan 0.985,
pero eso **no los iguala**: con 75 % de prevalencia casi cualquier cosa que se marque es un
fallo, y la métrica se satura.

## Umbral y política de decisión

El umbral no se elige por conveniencia:

- **Por costo.** Se barren razones de 1:1 a 100:1 entre el costo de no detectar y el de una
  inspección inútil. En la válvula el umbral **no se mueve en todo el rango**: la decisión
  es robusta. En el acumulador el óptimo manda inspeccionar el 98 %, que es correcto e
  inviable.
- **Por capacidad.** El umbral de inspección es el que marca el 15 % de los ciclos que el
  turno puede revisar.
- **Abstención.** Entre ambos queda la zona de vigilancia: ciclos que convendría revisar y
  no caben. No se declaran normales. Además, **todo ciclo sin régimen estacionario se
  abstiene siempre**.

## Limitaciones y riesgos

1. **El detector del acumulador depende del régimen.** Su recall es 0.009 con el
   refrigerador sano y **0.585 cuando el refrigerador está cerca del fallo total**. Solo ve
   el problema con la máquina ya caliente. Verificado que no es un artefacto: el acumulador
   está balanceado 120/120/120 entre los tres estados del refrigerador. **El de la válvula
   no tiene este problema**: recall 1.000 en todos los regímenes.
2. **La prevalencia del banco no es la de una planta.** 75 % de ciclos degradados porque es
   un experimento diseñado. Con la prevalencia real la precisión de la cola caería mucho.
3. **Una sola partición**, sin intervalos de confianza ni repetición con otras semillas.
4. **172 ciclos de entrenamiento.** El VAE opera cerca del límite de datos que necesita, lo
   que debilita, aunque no invalida, la conclusión sobre él.
5. **Marcar no es diagnosticar**, y una desviación no prueba causa.
6. **Datos de laboratorio.** No se traslada a producción sin volver a medir.

## Supervisión y monitoreo

Si se usara, vigilaría tres cosas: la **tasa de confirmación** de los ciclos inspeccionados,
que es la medida directa de si la cola sirve; la **proporción de ciclos en vigilancia**, que
si crece indica que la capacidad o el umbral están mal fijados; y la **estabilidad del
régimen**, porque el detector solo es válido dentro del que vio. Se retira si la tasa de
confirmación cae por debajo de lo acordado, y la priorización vuelve al orden por fecha.
