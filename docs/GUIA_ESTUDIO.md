# Guía del trabajo final · Detección de anomalías en un banco hidráulico

**INF-8239 Ciencia de Datos II · Tema 39 · VAE para detección de anomalías**
Alexis De La Cruz · 100805370

Fuente para estudio y defensa. Todas las cifras provienen de ejecutar los scripts del
repositorio `delaxis1/INF8239_TF`, etiqueta `tf-entrega`.

---

## 1. La pregunta, en una frase

¿Puede un sistema avisar de que un componente se está degradando **antes** de que una
aguja del panel entre en zona roja, y hace falta una red neuronal para conseguirlo?

### Por qué un panel de agujas es ciego

Un panel muestra una aguja por sensor, cada una con su banda verde y su zona roja. El
operador vigila **cada aguja contra su propio límite**. Eso le deja ciego ante la
**anomalía contextual**: cada lectura está dentro de rango, pero la combinación es
imposible.

El ejemplo que lo explica todo: 2800 revoluciones es normal. 78 °C es normal. **2800
revoluciones con 78 °C no lo es**, porque a esas revoluciones esa máquina siempre ha
trabajado a 62 °C. Ninguna aguja se ha movido de la zona verde, y sin embargo algo está
mal.

Detectar eso exige mirar la **relación entre variables**, no cada variable por separado.

---

## 2. Los datos

| | |
|---|---|
| Conjunto | Condition Monitoring of Hydraulic Systems · UCI 447 |
| Naturaleza | **Reales**, de un banco de pruebas instrumentado (ZeMA gGmbH) |
| Licencia | CC BY 4.0 · DOI 10.24432/C5CW21 |
| Sensores | 17: seis de presión, potencia del motor, dos de caudal, cuatro de temperatura, vibración y tres virtuales |
| Volumen | 2205 ciclos de 60 segundos · 43 680 lecturas por ciclo |
| Etiquetas | Degradación **por grados** en cuatro componentes |

La razón de elegir este conjunto y no uno de motores eléctricos con más sensores: para
detección de anomalías hacen falta **etiquetas de degradación**, y eso es lo raro. Los
conjuntos de motores públicos o no tienen fallos etiquetados, o son vibración cruda que
exige procesamiento de señal, o tienen variables anónimas.

---

## 3. Las cuatro trampas que encontró la auditoría de datos

Esta sección es la más importante para la defensa: son decisiones tomadas **antes** de
modelar, y cada una se apoya en una medición.

### Trampa 1 · Solo 21 ciclos están sanos en los cuatro componentes

De 2205 ciclos, apenas 21 tienen los cuatro componentes en estado óptimo. Con 21 muestras
no se entrena nada. **Decisión:** elegir un componente objetivo por vez y tratar los demás
como variación de fondo.

### Trampa 2 · El archivo está ordenado por el estado del refrigerador

El primer tercio del archivo son ciclos con el refrigerador al 3 %, el del medio al 20 %,
el último al 100 %. La correlación del índice con el estado es **+0.884**.

**Consecuencia:** una partición cronológica —la correcta en una serie temporal— aquí sería
catastrófica, porque entrenaría con un único régimen de refrigeración. **Decisión:**
partición estratificada aleatoria.

> **Punto de defensa.** La partición correcta depende de la estructura del dato, no de una
> regla memorizada. En el Ejercicio 02 la cronológica era la correcta; aquí es la
> equivocada, y el motivo está medido.

### Trampa 3 · El indicador de estabilidad delata la válvula

Los 1080 ciclos con la válvula degradada son **todos** estables. De los 1125 con válvula
óptima, solo 369 lo son.

**Consecuencia:** un modelo podría aprender «transitorio implica sano» y acertar 756 casos
sin entender nada. **Decisión:** quedarse solo con los 1449 ciclos estables, y una prueba
automática impide que ese indicador vuelva a entrar como variable.

### Trampa 4 · El refrigerador lo delata un solo sensor

| Sensor | Correlación con el estado del refrigerador |
|---|---|
| CE, eficiencia de refrigeración (**virtual**) | 0.992 |
| CP, potencia de refrigeración (**virtual**) | 0.956 |
| TS1, temperatura (real) | −0.911 |

**Consecuencia:** un operador *sí* vería ese fallo en el termómetro, así que no sirve para
la pregunta del trabajo. **Decisión:** descartar el refrigerador como objetivo.

---

## 4. Los objetivos elegidos

Ordenados por lo difícil que es detectarlos con una sola aguja:

| Objetivo | Correlación máxima de un sensor | Qué representa |
|---|---|---|
| **Válvula** | 0.231 | Anomalía contextual pura. Retardo de conmutación |
| **Fuga de bomba** | 0.469 | Caso intermedio. El componente lo acciona el motor eléctrico |
| **Acumulador** | 0.241 | El más complejo según la documentación del conjunto |

Los tres se evalúan con el mismo protocolo, de modo que la conclusión sobre el VAE no
depende de un objetivo concreto.

---

## 5. El hallazgo principal: la representación pesó más que el modelo

La válvula es un componente de **conmutación**. Sus niveles de degradación son «retardo
pequeño», «retardo severo» y «cerca del fallo total». Un retardo cambia **cuándo** sube la
presión dentro del ciclo de 60 segundos, **no su promedio**.

Resumir el ciclo con media, mínimo y máximo borra exactamente esa información:

| Representación | ROC-AUC |
|---|---|
| Media del ciclo por sensor | **0.58** |
| Seis tramos temporales por sensor | **0.996** |
| Doce tramos temporales | 1.000 |

Mismo modelo, mismos datos, mismo protocolo. **De casi azar a casi perfecto solo por
cambiar cómo se describe un ciclo.**

Este hallazgo llegó corrigiendo un error propio: la primera extracción de características
era la de la primera fila.

### La extracción definitiva

112 variables por ciclo: media y desviación de cada sensor, más la media de cada tramo
temporal — seis tramos para los nueve sensores rápidos de 100 y 10 Hz, y tres para los
ocho de 1 Hz, que solo tienen 60 lecturas por ciclo.

---

## 6. Las tres alternativas comparadas

| | Qué hace | Por qué está |
|---|---|---|
| **Alarma por aguja** *(baseline)* | Umbral por sensor, aprendido de lo normal. Alarma si alguna aguja sale de su banda | Es **lo que hace la planta hoy**. Comparar contra esto es honesto |
| **Mahalanobis** | Distancia a la nube de lo normal, teniendo en cuenta las correlaciones entre variables | El competidor justo del VAE. Si la relación es lineal, esto basta |
| **VAE** | Aprende una superficie que puede ser curva y multimodal. Puntúa por error de reconstrucción | Se gana el puesto solo si la relación **no** es lineal |

Los tres se entrenan **solo con ciclos normales** y fijan su umbral sobre normales de
validación, nunca sobre la prueba.

### Cómo funciona un VAE, en corto

El cuello de botella no es un vector sino una **distribución**: el codificador devuelve una
media y una varianza, se muestrea de ahí con el truco de reparametrización, y el
decodificador reconstruye. La pérdida tiene dos términos que tiran en direcciones
opuestas: el error de reconstrucción quiere copiar bien, y la divergencia KL obliga a que
el espacio latente se parezca a una normal. Esa tensión impide memorizar.

Para anomalías: se entrena **solo con lo normal**, y lo que no se parece a lo visto se
reconstruye mal. Ese error es la puntuación.

---

## 7. Resultados

Conjunto de prueba de 435 ciclos. Umbral al 5 % de falsas alarmas.
**La línea de azar del PR-AUC es la prevalencia de anomalías**: sin declararla, un 0.80
parece bueno y no lo es.

### Válvula · línea de azar 0.745

| | PR-AUC | ROC-AUC | Recall |
|---|---|---|---|
| Alarma por aguja | 0.8010 | 0.5512 | 0.142 |
| **Mahalanobis** | **0.9974** | **0.9934** | **1.000** |
| VAE | 0.9966 | 0.9912 | 0.870 |

### Acumulador · línea de azar 0.752

| | PR-AUC | ROC-AUC | Recall |
|---|---|---|---|
| Alarma por aguja | 0.8504 | 0.6160 | 0.352 |
| **Mahalanobis** | **0.9574** | **0.9115** | **0.456** |
| VAE | 0.9121 | 0.7829 | 0.284 |

Un ROC-AUC de 0.551 es **puntuar casi al azar**: el panel de agujas apenas distingue.

---

## 8. La detección temprana: lo que el trabajo quería medir

| Estado de la válvula | Panel de agujas | Mahalanobis | VAE |
|---|---|---|---|
| Retardo pequeño | **8.3 %** | **100 %** | 61.1 % |
| Retardo severo | 9.3 % | 100 % | 100 % |
| Cerca del fallo total | 25.0 % | 100 % | 100 % |

El operador ve 8 de cada 100 válvulas con retardo pequeño, y 25 de cada 100 cuando ya está
casi rota — es decir, **solo empieza a verlas cuando él mismo lo notaría**. El detector las
ve todas desde el primer nivel.

**Esa diferencia es la utilidad del sistema.**

---

## 9. ¿Es real la diferencia, o fue suerte?

Esta es la sección más importante para defender el trabajo, porque corrige una afirmación
que la primera versión del informe hacía de más.

El experimento se repitió con **diez particiones** sobre los **tres objetivos**, cambiando
solo qué ciclos caen en entrenamiento y cuáles en prueba.

| Objetivo | Gana Mahalanobis | Gana el VAE | Gana la alarma |
|---|---|---|---|
| Válvula | **7 de 10** | 3 de 10 | 0 |
| Fuga de bomba | **10 de 10** | 0 | 0 |
| Acumulador | **10 de 10** | 0 | 0 |

### Lo que NO se sostiene

En la válvula, los intervalos de confianza del PR-AUC **se solapan**: Mahalanobis
[0.9870, 0.9961] frente a VAE [0.9863, 0.9918]. Por esa métrica **la diferencia no es
distinguible del ruido**.

La medición con una sola partición —0.9974 frente a 0.9966— **sobreestimaba la
diferencia**. Medida una vez parecía una victoria; medida diez veces, no lo es.

### Lo que sí se sostiene, y con holgura

| Válvula | Recall medio | IC 95 % | Detección del retardo pequeño |
|---|---|---|---|
| **Mahalanobis** | **0.9997** | [0.9991, 1.0003] | **0.999 ± 0.003** |
| VAE | 0.8768 | [0.8049, 0.9488] | 0.631 ± **0.348** |
| Alarma por aguja | 0.1170 | [0.1034, 0.1305] | — |

**Esa desviación de 0.348 es el hallazgo.** El VAE no detecta peor en promedio: **detecta
de forma inestable**. En unas particiones ve el retardo pequeño y en otras casi no.
Mahalanobis tiene una desviación de 0.003.

### El argumento correcto

No es «Mahalanobis tiene mejor PR-AUC» —que en la válvula no se sostiene— sino
**«Mahalanobis es estable y el VAE no»**, que sí se sostiene en los tres objetivos.

Y la conclusión de fondo no cambia: la relación entre estos sensores es suficientemente
**lineal** como para que no haga falta una red neuronal. Un método de 1936 basta, con menos
parámetros y 2000 veces menos cómputo.

> **Punto de defensa.** Si preguntan por qué se corrigió el informe, la respuesta es que
> una sola partición no distingue «gana» de «ganó esta vez», y que medirlo diez veces era
> barato. Encontrarlo uno mismo antes de la defensa vale más que un número bonito.

---

## 10. Del puntaje a la decisión

### El umbral por costo

Una falsa alarma cuesta una inspección inútil. Un fallo no detectado cuesta una parada. La
relación entre ambos es decisión de la institución, no del modelo.

- **En la válvula el umbral no se mueve** entre razones de 1:1 y 100:1. La decisión es
  robusta: no hace falta conocer el costo con precisión.
- **En el acumulador el óptimo manda inspeccionar el 98 % de los ciclos.** Es
  matemáticamente correcto y operativamente imposible.

### La abstención, que sale de esa contradicción

Tres zonas: el umbral de inspección lo fija la **capacidad del turno**, y la **zona de
vigilancia** es la distancia entre lo que conviene revisar y lo que se puede revisar. Esos
ciclos no se declaran normales.

Además, **todo ciclo sin régimen estacionario se abstiene siempre**: el detector se entrenó
con ciclos estables y fuera de ese régimen no sabe lo que mira.

---

## 11. El límite que más importa

El detector del **acumulador solo funciona con la máquina ya caliente**:

| Estado del refrigerador | Recall del detector de acumulador |
|---|---|
| 100 % · eficiencia plena | **0.009** |
| 20 % · reducida | 0.019 |
| 3 % · cerca del fallo total | **0.585** |

Verificado que no es un artefacto: el acumulador está balanceado 120/120/120 entre los tres
estados del refrigerador.

Un sistema que solo ve el fallo cuando la máquina ya está degradada en otro sitio **no
sirve para anticiparlo**, que era el propósito. El de la válvula no tiene este problema:
recall 1.000 en todos los regímenes.

---

## 12. Lo que el trabajo NO demuestra

1. **La prevalencia del banco no es la de una planta.** Aquí el 74.5 % de los ciclos está
   degradado porque es un experimento diseñado. Con la prevalencia real, la precisión de la
   cola caería mucho.
2. **Marcar no es diagnosticar.** El sistema dice que un ciclo se aparta de lo normal, no
   qué componente falla ni por qué.
3. **Una desviación no prueba causa.** Indica dónde investigar.
4. **Son datos de laboratorio.** No se trasladan a producción sin volver a medir.

---

## 13. Preguntas probables en la defensa

**¿Qué evidencia justifica priorizar anomalías para investigación?**
La detección por nivel de severidad: el panel ve el 8 % de los retardos pequeños y el
detector el 100 %, con 2.7 % de falsas alarmas. Esa diferencia es tiempo de anticipación.

**¿Qué condición obligaría a detener, abstener o escalar?**
Abstención: régimen no estacionario, o puntuación en la banda entre el umbral de costo y el
de capacidad. Retiro: que la tasa de confirmación de las inspecciones caiga por debajo de
lo acordado; entonces la priorización vuelve al orden por fecha.

**¿Por qué no usaste validación cruzada?**
Se repitió con diez particiones y se publicaron las medias con sus intervalos de confianza.
Eso reveló que en la válvula la diferencia de PR-AUC estaba dentro del ruido, y obligó a
cambiar el argumento de «mejor PR-AUC» a «más estable». La corrección está documentada.

**¿Por qué el VAE, si al final no lo usas?**
Porque el tema pedía entrenarlo y usar su error de reconstrucción, y porque sin medirlo no
se puede afirmar que no hace falta. El resultado negativo **es** el resultado.

**¿Por qué no un conjunto de motores eléctricos?**
Porque para evaluar detección hacen falta etiquetas de degradación, y los conjuntos
públicos de motores o no las tienen, o son vibración cruda, o tienen variables anónimas.
El banco hidráulico tiene 17 sensores reales **y** degradación etiquetada por grados.

---

## 14. Glosario mínimo

| Término | Qué significa aquí |
|---|---|
| **Anomalía contextual** | Cada variable en rango, pero la combinación imposible |
| **PR-AUC** | Área bajo precisión-recall. Con clases desbalanceadas dice más que el ROC-AUC, y **siempre se compara contra la prevalencia** |
| **Línea de azar** | La prevalencia de anomalías: el PR-AUC que obtendría puntuar al azar |
| **Mahalanobis** | Distancia que tiene en cuenta las correlaciones entre variables |
| **Ledoit-Wolf** | Encogimiento de la covarianza, necesario cuando hay más variables que muestras |
| **Reparametrización** | Truco que permite muestrear en el VAE sin romper el gradiente |
| **Divergencia KL** | Término que obliga al espacio latente a parecerse a una normal |
| **Abstención** | Que el sistema diga «no sé» en lugar de emitir un veredicto poco fiable |
| **Sensor virtual** | Señal calculada, no medida. Aquí: CE, CP y SE |
