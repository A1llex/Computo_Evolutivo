# Tarea 2
## Resolver el siguiente problema:
### $f(x_1,x_2) = 418.9829*2 - x_1*\sin(sqrt( abs(x_1))) - x_2*\sin(sqrt(abs(x_2)))$ en rango $[-500, 500]$ para cada variable 

* **(40 puntos)** Escribir un programa en Python que implemente un genético simple con los operadores/representación asignados a cada quien y que retenga al mejor individuo en cada generación
* **(10 puntos)** El código debe imprimir en cada generación las siguientes estadísticas:
  * Máximo, media y mínimo de aptitud nominal
  * Número de cruzas y mutaciones efectuadas
  * Cadena del mejor individuo
* **(5 puntos)** Al final de la ejecución debera  Imprimir el 
  * mejor individuo encontrado, 
  * cadena, 
  * aptitud
* **(5 puntos)** Estime el tamaño del espacio de búsqueda considerando ***5 decimales*** de precisión e indique el porcentaje del espacio de búsqueda que exploró su algoritmo
* **(10 puntos)** Ejecución con parámetros mínimos ***6 individuos*** y ***2 generaciones*** donde se muestren la población (genotipo y fenotipo), los padres seleccionados, puntos de cruza, hijos, valores de mutación, etc.
* **(20 puntos)** Resultados promediados de ***20 ejecuciones*** del algoritmo. Soluciones mínima, media, máxima y desviación estándar. Usar mismos parámetros para las 20 ejecuciones y reportarlos. Se debe generar un archivo por cada ejecución del algoritmo
* **(10 puntos)** Reportar gráfica de convergencia. Eje x número de generaciones, eje y mediana de la mejor aptitud de cada generación
* **PDF** con los resultados de la ejecución mínima, resultados promediados, gráfica de convergencia y estimación de tamaño del espacio de búsqueda *(-30 puntos de no entregarlo).*

### Selección de componentes
**Alex Fernández** : 
- Representacion: Real Entera
- Selección de padres:  Universal Estocástica
- Escalamiento: Ninguno
- Cruza: Aritmética Total
- Mutación: Uniforme
- Selección: Más
