import tensorflow as tf
import pickle
import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn import svm
import functions

def main():
    args = functions.get_model_args()
    userId = args.id
    users_folder = f"{args.usersfolder}/User_{userId}"

    user_id_folder = os.path.join(users_folder, userId)  # Pot do mape z uporabnikovimi slikami
    person_does_not_exist_folder = os.path.join(users_folder, "PersonDoesNotExist")  # Pot do mape s slikami, kjer oseba ne obstaja

    matrikaUcna = functions.dodajtag(user_id_folder, person_does_not_exist_folder)
    oznake_ucne = [slika[1] for slika in matrikaUcna]

    matrika_značilk_ucne = functions.izlušči_značilnice(matrikaUcna)

    # Ustvarjanje modelov
    svm_model = svm.SVC(kernel='linear')

    # Učenje SVM modela na učni množici
    svm_model.fit(matrika_značilk_ucne, oznake_ucne)

    filename = f'../Model/{userId}.pkl' #model se poimenuje po idju od userja
    with open(filename, 'wb') as f:
        pickle.dump(svm_model, f)

if __name__ == "__main__":
    main()
