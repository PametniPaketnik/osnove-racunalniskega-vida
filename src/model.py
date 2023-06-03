import tensorflow as tf
import pickle
import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn import svm
import functions

userId = "646d0c1fa7d8e13e080352d7"  # dobimo iz app
users_folder = f"../images/TrainImages/User_{userId}"  # Pot do glavne mape z uporabniki

user_id_folder = os.path.join(users_folder, userId)  # Pot do mape z uporabnikovimi slikami
person_does_not_exist_folder = os.path.join(users_folder, "PersonDoesNotExist")  # Pot do mape s slikami, kjer oseba ne obstaja

matrikaUcna = functions.dodajtag(user_id_folder, person_does_not_exist_folder)
oznake_ucne = [slika[1] for slika in matrikaUcna]
# print(oznake_ucne)

'''
for slika in matrikaUcna:
    print(slika[0], slika[1])
'''

matrika_značilk_ucne = functions.izlušči_značilnice(matrikaUcna)
#print(matrika_značilk_ucne)

# Ustvarjanje modelov
svm_model = svm.SVC(kernel='linear')

# Učenje SVM modela na učni množici
svm_model.fit(matrika_značilk_ucne, oznake_ucne)

filename = f'../Model/{userId}.pkl' #model se poimenuje po idju od userja
with open(filename, 'wb') as f:
    pickle.dump(svm_model, f)
