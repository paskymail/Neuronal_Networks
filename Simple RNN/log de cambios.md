# Reuniones de equipo
## Lunes 20/04
### Resultados
       1. La RNN entrenada como problema de regresión da los mismos resultados que como problema de clasificación
       2. La RNN multi step con retorno de secuencia da mejores resultados.
### Acciones
        1. Generar caminos más largos para alimentar la RNN
        2. Explicar comportamiento del retorno de secuencia
             - Entropía cruzada
            - Random forest 
        3. Añadir padding
   

## Lunes 27/04
### Resultados
       1. Tras analizar el random forest, todas las variables se utilizan de forma parecida
       2. Caminos más largos no generan mejores predicciones
### Acciones
        1. Usar un encoder-decoder para predecir N pasos
        2. Predecir la misma secuencia decalada de una unidad para predecir un paso
        3. Añadir embeddings para codificar las acciones

## Lunes 04/05
### Resultados
       1. El uso de la sequencia decalada mejora la precisión y evita el sobre aprendizaje para un mismo número de parámetros
       2. Random forest: confusión selectiva entre ciertas acciones
### Acciones
        1. Usar un encoder-decoder para predecir N pasos
        2. Añadir embeddings para codificar las acciones
        3. Redactar resultados de secuencial decalada usando information theory


RNN con secuencia decalada
1 -> 2
2 -> 3
3 -> 4

F(X) --> entrada
F´(y) --> salida

RNN Actual
1-> 3
2-> 4

1, 2, 3 ,4 --> 5

Encoder-decoder



Embedding
esto 1 (1 0 0 0 0 0)  (0.1 0.3 0.5)
ha
sido 2 (0 1 0 0 0 0) (0.12 0.4 0.45)
un 3 (0 0 1 0 0 0) (0.12 0.4 0.45)
gato 4  

this 
was
a 
cat

Esto 1
es 2
un 5 (0.12 0.42 0.35)
perro 6
