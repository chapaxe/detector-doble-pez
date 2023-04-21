# detector-doble-pez
proyecto simple de deep learning, consiste en un usar un clasificador en un video de linea de produccion en base a un proyecto anterior.

el modelo (modelo_cnn_peces.h5) clasifica entre peces simple, pez doble o paleta (empujadora).

El programa parte de la base de un segidor de objetos cualquiera, este se usa para detectar y segir los peces a medida que se mueven en una linea de producion
pero en cada frame se ejecuta el clasificador (y se ejecuta de manera muy mal optimizada) para detectar si los objetos que se estan siguiendo es un pez doble o no, si es un pez doble entonces se resalta con un rectangulo rojo.

Como ya se menciono, este programa esta muy mal optimizado, lo desarrolle solamente para ver la ejecucion del modelo clasificador de peces.

output.AVI es el video inicial (se llama output porque es un recorte de un video mas largo de la linea de produccion)
deteccion_doblepez.AVI es un video que muestra el resultado.

se puede hacer que el programa grabe un video descomentando las lineas 137, 138, 253 y 267

El Progama requiere openCV 4.7.0 y Tensorflow
