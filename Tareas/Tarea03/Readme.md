# Tarea 3
## Desarrollar un algoritmo evolutivo en Python con las siguientes características:


- **(20 puntos)** Seleccionar los componentes básicos del evolutivo
- Seleccionar al menos dos técnicas avanzadas
    - Paralelismo
    - Coevolución
    - Metamodelos
    - Técnicas de diversidad
    - Meméticos
    - Hiper heurísticas 

- **(20 puntos)** Aplicar el evolutivo a los siguientes 5 problemas con n=10 (https://www.sfu.ca/~ssurjano/optimization.html)
    - Rastrigin
    - Ackley
    - Rosenbrock
    - Eggholder
    - Easom

- **(20 puntos)** Realizar un estudio comparativo entre el algoritmo desarrollado en la tarea 2 y el diseñado en está tarea utilizando hasta 10mil evaluaciones de función utilizando la prueba de Wilcoxon rank sum
- **(10 puntos)** Los parametros del nuevo evolutivo deben estar optimizados con iRace
- **(20 puntos)** Resultados promediados de 20 ejecuciones del algoritmo. Soluciones mínima, media, máxima y desviación estándar. Usar mismos parámetros para las 20 ejecuciones y reportarlos. Se debe generar un archivo por cada ejecución del algoritmo

- **(10 puntos)** Reportar gráfica de convergencia. Eje x número de generaciones, eje y mediana de la mejor aptitud de cada generación
- PDF con los resultados de la ejecución mínima, resultados promediados, gráfica de convergencia y estimación de tamaño del espacio de búsqueda **(-30 puntos de no entregarlo)**.

### Selección de componentes
**Alex Fernández** : 
- Representacion: Real Entera
- Selección de padres:  Universal Estocástica
- Cruza: Aritmética Total
- Mutación: Uniforme
- Selección: Más
### Tecmica Avanzada
- Meméticos : Lamarkiano con decenso del gradiente
- Tecnica de diversidad : Procedimiento de clearing
