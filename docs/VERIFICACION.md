# Verificación de citas, licencias y reproducibilidad

**INF-8239 Ciencia de Datos II · Trabajo final · Tema 39** · Alexis De La Cruz · 100805370

La escala de la rúbrica califica «insuficiente» cualquier afirmación sin evidencia o
artefacto no verificable. Este documento recoge qué se comprobó, cómo y qué salió mal.

## 1. Citas y licencias

| Afirmación | Fuente consultada | Resultado |
|---|---|---|
| Conjunto bajo licencia **CC BY 4.0** | Página de UCI 447 | ✅ Verbatim: *«This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license»* |
| DOI del conjunto **10.24432/C5CW21** | Página de UCI 447 | ✅ Coincide |
| Donado en **2018** por ZeMA gGmbH | Página de UCI 447 y `documentation.txt` | ✅ Coincide |
| **17 sensores**, ciclos de 60 s, 2205 instancias | `documentation.txt` del propio conjunto | ✅ Coincide |
| DOI del artículo **10.1109/I2MTC.2015.7151267** | Resolución del DOI y registros bibliográficos | ✅ Coincide |
| Páginas del artículo | Registros bibliográficos del I2MTC 2015 | ❌ **Corregido** |

### El error encontrado

La cita del artículo de Helwig decía **páginas 1-6**. Las páginas reales son **210-215**.

El «1-6» no salía de ninguna fuente: es la convención habitual de numeración en actas de
congresos del IEEE, y se escribió por inercia sin comprobarlo. La documentación del propio
conjunto cita el artículo como *«paper PPS1-39, Pisa, Italy, May 11-14, 2015»*, sin páginas,
de modo que la referencia original tampoco las aportaba.

Corregido en `README.md` y en el generador del informe. **Es el único error que encontró
esta verificación, y es exactamente el tipo de cosa para la que se abrió.**

## 2. Reproducibilidad

Procedimiento: clonar el repositorio en una carpeta vacía, instalar desde cero con `uv
sync`, descargar los datos y ejecutar los cuatro scripts. Después, comparar **archivo por
archivo** contra la copia de trabajo.

| | |
|---|---|
| Huella SHA-256 del zip descargado | ✅ Idéntica |
| Huellas de los 18 archivos de datos | ✅ **18 de 18 idénticas** |
| Pruebas en la copia limpia | ✅ Pasan |
| Archivos de `reports/` comparados | 55 |
| **Idénticos byte a byte** | **49** |
| Difieren | 6 |

### Qué difiere, y por qué no importa

Los seis archivos que difieren lo hacen **únicamente en columnas de tiempo de reloj**:
`ajuste_s` y `latencia_ms_por_ciclo`. Por ejemplo, el ajuste del VAE sobre la válvula pasa
de 7.97 s a 13.70 s según la carga de la máquina.

**Ninguna métrica cambia.** PR-AUC, ROC-AUC, recall, falsas alarmas, umbrales, número de
parámetros y tamaño del modelo coinciden hasta el último dígito en los cuatro experimentos
y en las dos capas de decisión. El entrenamiento del VAE es determinista: su tiempo varía,
sus resultados no.

Que las cifras de tiempo no se repitan es lo correcto —son medidas físicas de una máquina
compartida— y por eso el informe las presenta como órdenes de magnitud y no como cifras
exactas.

## 3. Ninguna cifra escrita a mano

`tests/test_documentacion.py` lee las cifras de `reports/` y comprueba que aparecen
literalmente en el `README.md` y en el `MODEL_CARD.md`: PR-AUC y ROC-AUC de las tres
alternativas en los objetivos evaluados, la línea de azar, la detección por nivel de severidad y
el recall por régimen de refrigeración.

Si alguien vuelve a ejecutar un script y una cifra cambia, **la suite falla** en lugar de
dejar una documentación que miente. Eso es lo que convierte «sin cifras a mano» de promesa
en propiedad comprobada.

> **Alcance de esta comprobación.** Se hizo sobre la versión etiquetada `tf-entrega`, con
> dos objetivos y 55 archivos en `reports/`. Las versiones posteriores añaden un tercer
> objetivo, diez particiones, el caso trabajado y las figuras, de modo que hoy `reports/`
> tiene más archivos. El procedimiento es el mismo y los scripts siguen siendo
> deterministas; lo que no se ha repetido es la comparación byte a byte completa.

## 4. Lo que esta verificación no cubre

- **No se consultó IEEE Xplore directamente** para las páginas del artículo: el acceso
  requiere suscripción. La corrección se apoya en registros bibliográficos coincidentes,
  no en el PDF del editor.
- **La reproducibilidad se comprobó en la misma máquina.** Otro sistema operativo o otra
  versión de BLAS podrían mover los últimos decimales.
- **El conjunto se descargó dos veces del mismo servidor**, así que la coincidencia de
  huellas prueba que el archivo no cambió, no que sea el mismo que usó el autor original.
