# Declaración de uso de asistentes de IA

**INF-8239 Ciencia de Datos II · Trabajo final · Tema 39** · Alexis De La Cruz · 100805370

El mandato permite que la IA apoye estructura, pruebas o redacción, pero **no que decida
los valores del análisis ni invente evidencia**. Esta declaración detalla dónde estuvo la
línea.

## Herramienta

**Claude Opus 5 (Anthropic)**, mediante Claude Code, con acceso al proyecto local y al aula
virtual para leer el mandato, la rúbrica y la ficha del Tema 39.

## Instrucciones relevantes

Las instrucciones se recogen **resumidas en su contenido, no transcritas literalmente**.
El orden es el real y cada una corresponde a un punto en que el trabajo cambió de rumbo.

| Instrucción dada al asistente | Qué determinó |
|---|---|
| Selección del Tema 39 y lectura del mandato, la rúbrica y los criterios de aceptación | Alcance y entregables |
| **Planteamiento del problema:** detectar degradación de equipos a partir de la relación entre revoluciones, temperaturas y vibraciones, partiendo de que un operador solo advierte el fallo cuando una variable entra en rango rojo | **La tesis completa del trabajo.** Es mío y de aquí sale todo lo demás |
| Preferencia por un entorno industrial más instrumentado y con más sensores | Búsqueda y verificación de conjuntos candidatos en UCI |
| Instrucción de **anteponer la calidad del conjunto al tiempo de preparación**, aceptando dedicar dos días a la ingeniería de datos | **Mi decisión** de trabajar con datos reales en lugar de sintéticos |
| Selección de la **válvula** como componente objetivo, sobre la evidencia univariante presentada | **Mi decisión** del objetivo principal |
| Incorporación del **acumulador** como segundo caso, de dificultad mayor | **Mi decisión** de ampliar el alcance por encima del mínimo |
| Decisión sobre la visibilidad del repositorio, tras discutir los riesgos | **Mi decisión** de publicarlo |

## Qué decidí yo

| Decisión | Quién |
|---|---|
| El planteamiento: anomalía contextual invisible en una aguja individual | **Yo** |
| Conjunto real del banco hidráulico frente al sintético, asumiendo su costo | **Yo** |
| Componente objetivo: válvula, y después el acumulador como caso difícil | **Yo** |
| Alcance por encima del mínimo individual | **Yo** |
| Visibilidad del repositorio | **Yo** |
| Cifras del análisis | Ninguno de los dos: salen de ejecutar los scripts |

El asistente me planteó explícitamente las decisiones en lugar de rellenarlas, y cuando la
evidencia contradijo algo ya escrito me lo dijo antes de seguir.

## Qué hizo el asistente

- Verificó los conjuntos candidatos en UCI: licencia, tamaño, estructura y limitaciones.
- Escribió el código: descarga con huellas, extracción, baseline, Mahalanobis, VAE,
  evaluación, capa de decisión y tablero.
- Ejecutó los experimentos y ordenó la evidencia en `reports/`.
- Redactó los borradores del README, el Model Card y esta declaración.
- Escribió las 38 pruebas.

## Qué verifiqué

- **Todas las cifras provienen de ejecutar los scripts** en este equipo. No hay ningún
  número escrito a mano en el README ni en el Model Card.
- **Las huellas SHA-256** del zip y de los 18 archivos las calcula el propio script.
- **Que el hallazgo de la representación es real**: comparé 1, 6, 12 y 30 tramos
  temporales antes de rediseñar nada.
- **Que la dependencia del régimen del acumulador no es un artefacto**: comprobé la tabla
  cruzada y está balanceado 120/120/120 entre los tres estados del refrigerador.
- **Que las contribuciones del tablero suman la distancia**, con una prueba automática.
- **Las 38 pruebas** pasan con `uv run pytest -q`.

## Correcciones durante el trabajo

Las cinco que cambiaron el resultado, no erratas:

1. **El diseño inicial no era viable.** Entrenar con ciclos sanos en los cuatro
   componentes dejaba 21 muestras de 2205. Se pasó a un componente objetivo por vez.
2. **El refrigerador se descartó después de elegirlo.** Parecía el candidato obvio hasta
   medir que CE lo predice con 0.992 y TS1 con −0.911: un operador lo vería en una sola
   aguja, así que no servía para la pregunta.
3. **La partición cronológica habría sido un error aquí.** El archivo está ordenado por
   estado del refrigerador, con correlación del índice +0.884. Es lo contrario de lo que
   convino en los Ejercicios 01 y 02, y por eso se justifica en el código.
4. **La extracción de características estaba mal y hubo que rehacerla.** Los agregados del
   ciclo borraban el retardo de conmutación, que es donde vive el estado de la válvula:
   ROC-AUC 0.58 frente a 0.996 con tramos temporales. Fue el hallazgo principal del
   trabajo y llegó corrigiendo un error propio.
5. **Al VAE se le dio una segunda oportunidad antes de declararlo perdedor.** La primera
   ejecución agotó las 400 épocas sin converger. Se repitió con parada temprana hasta 3000
   épocas: mejoró de 0.333 a 0.611 en el nivel leve y siguió perdiendo. La conclusión se
   escribió solo después de eso.

Dos erratas menores corregidas: el umbral de Mahalanobis se estaba aplicando a los tres
detectores en la tabla Green AI pese a tener escalas distintas, y la banda de abstención
se definía por percentil, lo que la dejaba vacía o inconsistente según el objetivo. Ambas
se detectaron al revisar resultados que no cuadraban.

## Responsabilidad

El código, los datos, las referencias, la interpretación y las conclusiones son
responsabilidad mía. Puedo explicar cualquier fragmento del proyecto y defender cada
decisión, empezando por la que ordena el trabajo: por qué un panel de agujas no puede ver
una combinación imposible de lecturas normales.
