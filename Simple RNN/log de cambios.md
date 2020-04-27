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
        1. Usar un encoder decoder para predecir N pasos
        2. Predecir la misma secuencia decalada de una unidad para predecir un paso
        3. Añadir embeddings para codificar las acciones
        4. adsf
        5. adsf
        6.  
         
