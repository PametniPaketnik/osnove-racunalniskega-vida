import pickle
import cv2
import os
import argparse
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn import svm
import functions

def main():
    args = functions.get_login_args()

    if args.id is None or args.imgpath is None or args.outputpath is None:
        print("Problem z argumenti")
        return None
    
    #current_path = os.getcwd()
    #directory = os.path.dirname(os.path.dirname(current_path))
    #directory = f"{directory}/osnove-racunalniskega-vida"
    haarcascade_path = "haarcascade_frontalface_default.xml"
    #print(haarcascade_path)

    image_path = args.imgpath.strip("'")
    #print(image_path)
    output_path = args.outputpath.strip("'")
    #print(output_path)
    #print("mjaw")

    # Naloži sliko
    image = cv2.imread(image_path)

    # Ustvari detektor obrazov
    try:
        # Pretvori sliko v sivinsko obliko
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #print(gray)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        #if face_cascade.empty():
            # print("Kaskadni klasifikator ni bil uspešno naložen.")
        #else:
            # print("Kaskadni klasifikator je bil uspešno naložen.")

        # Zazna obraz na sliki
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20,)
    except Exception as e:
        # Handling the exception and accessing the error message
        error_message = str(e)
        print("An error occurred:", error_message)
        return None

    # Preveri, ali je bil obraz zaznan
    if len(faces) > 0: # Izreži prvi zaznani obraz
        (x, y, w, h) = faces[0]
        face_image = image[y:y+h, x:x+w]

        # Shrani izrezan obraz na izhodno mesto
        cv2.imwrite(output_path, face_image)
        # print("Obraz je bil shranjen.")
    else:
        print("Na sliki ni bilo zaznanega obraza.")
        return None

    matrikaSlika = []
    ciljna_velikost = (400, 400)

    user_images = [output_path]  # Seznam poti do slik

    for image_path in user_images:
        image_name = os.path.basename(image_path)  # Dobimo ime slike
        slika = cv2.imread(image_path)

        if slika is not None:
            obrezana_slika = cv2.resize(slika, ciljna_velikost)
            matrikaSlika.append([obrezana_slika, 0])
            # print(f"Slika {image_name} je bila uspešno dodana v matriko.")
        else:
            print(f"Napaka pri nalaganju slike {image_name}.")
            return None

    značilnice = [slika[0] for slika in matrikaSlika]
    oznake_ucne = [slika[1] for slika in matrikaSlika]
    # print(značilnice)

    matrika_značilk_ucne = functions.izlušči_značilnice(matrikaSlika)

    # Branje modela iz datoteke
    model_path = f"D:/FERI/2_letnik/2_semester/paketnik/osnove-racunalniskega-vida/{args.id}.pkl"
    #print(args.id)

    try:
        with open(model_path, "rb") as file:
            svm_model = pickle.load(file)
    except:
        print("Model ne obstaja")
        return None

    # Napovedovanje z modelom
    napovedi = svm_model.predict(matrika_značilk_ucne)

    # Primerjava napovedi z oznakami
    for napoved, oznaka in zip(napovedi, oznake_ucne):
        if napoved == oznaka:
            print(True)
        else:
            print("Napoved je napačna.")
            print(False)

if __name__ == "__main__":
    main()
