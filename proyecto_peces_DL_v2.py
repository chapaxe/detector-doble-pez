# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:53:31 2023

@author: pizza
"""

from __future__ import print_function
import cv2
import numpy as np
import argparse
from scipy.spatial import distance
import tensorflow
from tensorflow.keras.models import load_model



#se define una clase mancha para el segimiento de objetos
class mancha:
    def __init__(self, _contours, centro_inicial):
        self.nombre= 0
        self.contorno= _contours
        self.fue_contado= False
        self.es_mancha_anterior= False
        self.aun_siguiendo= True
        self.frames_sin_match = 0
        self.centro=[]
        self.centro.append(centro_inicial)
        
        self.prediccion= centro_inicial
       
# el metodo predecir de la clase mancha predice la proxima posicion (o la posicion mas cercana)
# que el objeto mancha deberia tener en el proximo frame
    def Predecir(self):
        x= self.centro[-1][0]
        y= self.centro[-1][1]
        if (x>=415):
            self.prediccion=[x-5,y+5]
        else:
            self.prediccion=[x+5,y+5]


#esta funcion encuentra une los objetos que se estan siguiendo en el frame actual
# con los objetos de los frames anteriores
def match_BlobFrameActualToBlobExistentes(existingBlobs, currentBlobs):
    for i in existingBlobs:
        i.es_mancha_anterior=False
        i.Predecir()
    
    for i in currentBlobs:
        distancia_min=150
        indice_dist_min=0
        for ind in range(len(existingBlobs)):
            if (existingBlobs[ind].aun_siguiendo==True):
                # hacer el calculo de la distancia_euc aqui
                
                #print(i.centro[-1], existingBlobs[ind].centro[0])
                
                distancia = distance.euclidean(i.centro[-1], existingBlobs[ind].centro[0])
                
                if distancia<distancia_min:
                    distancia_min = distancia
                    indice_dist_min = ind
                    
        #se se cumplen las condiciones, es un match, por lo que se actualizan
        # los datos del objeto con los datos del frame actual
        if (i.centro[-1][1] <= 135)&(distancia_min < 15)&(distancia_min > 0):
 
            addBlobToExistingBlobs(existingBlobs,i,indice_dist_min)
        elif(i.centro[-1][1] in range(136,270))&(distancia_min < 25)&(distancia_min > 0):

            addBlobToExistingBlobs(existingBlobs,i,indice_dist_min)
        elif(i.centro[-1][1] in range(271,400))&(distancia_min < 25)&(distancia_min > 0):

            addBlobToExistingBlobs(existingBlobs,i,indice_dist_min)
        elif(i.centro[-1][1]>400)&(distancia_min<40)&(distancia_min > 0):

            addBlobToExistingBlobs(existingBlobs,i,indice_dist_min)
        else:
              #si no se matchea, entonces agregar la mancha a las manchas existentes
              i.frames_sin_match=0
              existingBlobs.append(i)
        
    for z in existingBlobs:
        if not z.es_mancha_anterior:
            f_count=z.frames_sin_match
            z.frames_sin_match=f_count + 1            
        if z.frames_sin_match>3:
            z.aun_siguiendo= False
        
        
# esta funcion actualiza los datos del objeto a segir con sus datos correspondientes
# al frame actual
def addBlobToExistingBlobs(existingBlobs, currentBlob, indice):
    existingBlobs[indice].contorno = currentBlob.contorno
    existingBlobs[indice].centro=currentBlob.centro
    existingBlobs[indice].frames_sin_match=0
    existingBlobs[indice].aun_siguiendo=True
    existingBlobs[indice].es_mancha_anterior=True

    
 # dibuja un rectangulo en lo que se quiera resaltar
def dibuja_manchas(man_Corr,masked_frame):
    # print('iniciando dibujos')
    for x in man_Corr:
        # print(cv2.contourArea(x.contorno))
        # cv2.putText(masked,str(cv2.contourArea(x.contorno)),x.centro[-1],cv2.FONT_HERSHEY_SIMPLEX,0.25,(0, 0,255),1)
                
        if x.aun_siguiendo :
            x,y,w,h = cv2.boundingRect(x.contorno)
            cv2.rectangle(masked,(x-3,y-3),(x+w+3,y+h+3),(0,255,0),2)

    # print('terminado dibujos')



#se empieza inicializando el algoritmo de background substraction
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()
 

# se carga el modelo del clasificador
model=load_model('modelo_CNN_peces.h5')


# se inicia la captura del video

#cap = cv2.VideoCapture('chacamanca primer linea.AVI')
cap = cv2.VideoCapture('output.avi')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('deteccion_test.AVI', fourcc, 14.31, (800,600))
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
  
  
manchas_actuales=[]
manchas_anteriores=[]
pop_cherry=1

while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
 
    # enmascarar el frame para dejar solo la linea de proceso:
        
    img = np.zeros(frame.shape[:2], dtype="uint8")
    puntos = np.array([[370,40],[310, 599],[393, 599],[396, 40]]).reshape(1, -1, 2)
    cv2.fillPoly(img, pts=puntos, color=(255,255,255))
    pts2 = np.array([[426,40],[498, 599],[575,599],[450,40]], np.int32).reshape(1, -1, 2)
    cv2.fillPoly(img, pts2, (255,255,255))
    #cv2.imshow('mascara', img)
    
    #se aplica la mascara y el metodo de subtraccion de fondo
    masked = cv2.bitwise_and(frame, frame, mask=img)
    fgMask = backSub.apply(masked) 
    ret2,transf1= cv2.threshold(	fgMask, 126, 255, cv2.THRESH_BINARY	)          
    transf=cv2.GaussianBlur(	transf1, (5, 5), 0	)
    ret2,transf2= cv2.threshold(	transf, 126, 255, cv2.THRESH_BINARY	)
    cv2.imshow('FG Mask', transf2)
    
    #encontrar contornos, aprovechando que lo unico en escena son 
    # los peces moviendose a traves de la linea
    contours, hierarchy =	cv2.findContours( transf2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img2 = np.zeros(frame.shape[:2], dtype = "uint8")
    cv2.drawContours(img2, contours, -1, (255,255,255),-1)

    img2=cv2.morphologyEx(img2, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ( 2*4 + 1, 2*4+1 )))
    img2=cv2.morphologyEx(img2, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ( 2*2 , 2*2+2 )))
    cv2.imshow('pantalla negra',img2)
    
    contours, hierarchy =	cv2.findContours( transf2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours))

 
    """
ahora viene la parte de pasar de contornos a tipo manchas
    """
 
    for blobs in contours:
        M = cv2.moments(blobs)
        area = cv2.contourArea(blobs)
        
        if (M['m00'] !=0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx=0
            cy=0
        
        # se discriminan las manchas dependiendo de la posicion y el tamaño para evitar ruido
        # hecho esto se agregan al listado de manchas actuales
        if (area >40)&(area<4500)&(cy<150):
            pez_ini = mancha(blobs,[cx,cy])
            manchas_actuales.append(pez_ini)
        elif (area>200)&(cy>150):
            pez_ini = mancha(blobs,[cx,cy])
            manchas_actuales.append(pez_ini)
            
    # aqui se agrego una condicion inicial para evitar comparar en el primer frame
    # ya que se estaria comparando con nada, en vez de eso se espera
    # que se llene el listado de manchas actuales para empezar a comparar        
    if(pop_cherry) & (len(manchas_actuales)>10):
        manchas_anteriores.extend(manchas_actuales)
        pop_cherry=0
        print('pop the cherry')
        print(len(manchas_actuales))
    else:
        #se comparan los listados de manchas para empeza el tracking
        match_BlobFrameActualToBlobExistentes(manchas_anteriores, manchas_actuales)

    manchas_actuales.clear()

    
    ### aqui va la rutina para limpiar las manchas anteriores
    # se eliminan los elementos que ya no estan presentes en escena
    new_list = [item for item in manchas_anteriores if item.aun_siguiendo]
    manchas_anteriores.clear()
    manchas_anteriores= new_list             
    
    # esta es la rutina de clasificadora, se usan los contornos de los 
    # objetos que se estan siguiendo para determinar un area de interes
    # usamos esta area de interes para ejecutar el clasificador, el 
    # cual dira si es un pez simple,un pez doble o una paleta (que empuja)
    
    for x in manchas_anteriores:
        z,y,w,h = cv2.boundingRect(x.contorno)
        # roi = np.zeros((h,w,3), dtype="uint8")
        roi = masked[y:y+h, z:z+w]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        image_shape = (25,44) 
        resized = cv2.resize(rgb_roi, image_shape, interpolation = cv2.INTER_AREA) 
        resized = np.expand_dims(resized, axis=0)
        predictions = model.predict(resized)
        
        #si el area de interes corresponde a un pez doble, se dibujara un rectangulo para resaltarlo
        if predictions[0][0]==1:
            # print("pez doble")
            cv2.rectangle(frame,(z-3,y-3),(z+w+3,y+h+3),(0,0,255),2)
    
    cv2.imshow('Frame',frame)
    # out.write(frame)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
    
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object

cap.release()
#out.release()
 
# Closes all the frames
cv2.destroyAllWindows()