# Proyecto-EL4106: "Colorización automática de imágenes en escala de grises"
#### Integrantes: Diego Canales y Juan Francisco Torrejón
#### Profesor: Pablo Estévez
#### Auxiliar: Ignacio Reyes
#### Tutor: Germán García

Este proyecto implementa modelos que apuntan a solucionar el problema de colorización automática de imágenes en escala de grises. A continuación se muestran instrucciones para ejecutar el código correspondiente al primer modelo propuesto como un acercamiento a la solución del problema.

Para entrenar el modelo, en primer lugar debe ejecutarse desde algún IDE de *Python* el archivo **modelo.py**, que se encarga de implementar la arquitectura del modelo y también las funciones necesarias para los procesos de *training* y *testing*.

Una vez ejecutado dicho archivo, desde llamarse a la funcion **train_(data_size, img_size)** que recibe como parámetros el tamaño deseado para el conjunto de entrenamiento y el tamaño con que se desea trabajar las imágenes . En nuestro caso, el tamaño del *training set* que se utilizó fue 1000 y 224 como tamaño de las imágenes. Esto iniciará el entrenamiento del modelo y lo guardará la configuración y los pesos calculados:
```sh
train_(1000, 224)
```
Luego, para realizar pruebas, se llama a la función **test_compare(img_size)** que recibe nuevamente el tamaño de la imagen. La implementación de esta función asume que el entrenamiento fue ejecutado en dos modelos distintos, guardando los modelos y los pesos calculados en archivos respectivos, que en este caso se encuentran en el repositorio. Selecciona aleatoriamente una imagen desde el conjunto de pruebas y la muestra junto a su versión en escala de grises y a los resultados entregados por los modelos:
```sh
test_compare(224)
```
Ejemplos de resultado de esta ejecución se muestran en la figura a continuación:

![Ejemplos de *test*](https://github.com/DiegoCanalesR/Proyecto-EL4106/blob/master/prelim.PNG)

Con respecto a las funciones implementadas en **preprocess.py**, no es necesario ejecutar manualmente este programa, puesto que todas las funcionalidades quedan implícitas dentro de **modelo.py**.