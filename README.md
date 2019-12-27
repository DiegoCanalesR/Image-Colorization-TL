# Proyecto-EL4106: "Colorización automática de imágenes en escala de grises"
#### Integrantes: Diego Canales y Juan Francisco Torrejón
#### Profesor: Pablo Estévez
#### Auxiliar: Ignacio Reyes
#### Tutor: Germán García

Este proyecto implementa modelos que apuntan a solucionar el problema de colorización automática de imágenes en escala de grises. A continuación se muestran instrucciones para ejecutar el código correspondiente a los distintos modelos propuestos como un acercamiento a la solución del problema. Las distintas arquitecturas y detalles sobre los modelos pueden encontrarse en **Informe.pdf**.

Para entrenar el modelo, en primer lugar debe ejecutarse desde algún IDE de *Python* el archivo correspondiente a cada uno de los modelos:
1. **modelo_1_mse.py** para  el primer modelo utilizando *loss* MSE.
2. **modelo_2_mse.py** para el segundo modelo utilizando *loss* MSE.
3. **modelo_1_lossprob.py** para  el primer modelo utilizando *loss* probabilística.
4. **modelo_2_lossprob.py** para el segundo modelo utilizando *loss* probabilística.


Cada programa de la lista se encarga de implementar la arquitectura de un modelo y también las funciones necesarias para los procesos de *training* y *testing*.

Una vez ejecutado alguno de los archivs, desde llamarse a la funcion **train_model()**. Esto iniciará el entrenamiento del modelo y guardará la configuración y los pesos calculados:
```sh
train_model()
```
Luego, para realizar pruebas, se llama a la función **test_model()** que recibe nuevamente el tamaño de la imagen. Selecciona aleatoriamente una imagen desde el conjunto de pruebas y la muestra junto a su versión en escala de grises y al resultado entregado por el modelo entrenado:
```sh
test_model()
```
Ejemplos de resultado de esta ejecución se muestran en la figura a continuación:

Ejemplos obtenidos con el primer modelo con *loss* MSE:

![Ejemplos de *test*](https://github.com/DiegoCanalesR/Proyecto-EL4106/blob/master/res1mse.PNG)

Ejemplos obtenidos con el segundo modelo con *loss* MSE:

![Ejemplos de *test*](https://github.com/DiegoCanalesR/Proyecto-EL4106/blob/master/res2mse.PNG)

Ejemplos obtenidos con el primer modelo con *loss* probabilístico:

![Ejemplos de *test*](https://github.com/DiegoCanalesR/Proyecto-EL4106/blob/master/res1prob.PNG)

Ejemplos obtenidos con el segundo modelo con *loss* probabilístico:

![Ejemplos de *test*](https://github.com/DiegoCanalesR/Proyecto-EL4106/blob/master/res2prob.PNG)

Con respecto a las funciones implementadas en **preprocesamiento.py**, no es necesario ejecutar manualmente este programa, puesto que todas las funcionalidades quedan implícitas dentro de **herramientas.py**.
