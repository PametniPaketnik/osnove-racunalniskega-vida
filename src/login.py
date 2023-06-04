import pickle
import cv2
import os
import argparse
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn import svm
import functions

def get_args():
    parser = parser = argparse.ArgumentParser()

    parser.add_argument('--id', help='ID name')
    parser.add_argument('--imgpath', help='ID name')
    parser.add_argument('--outputpath', help='ID name')

    return parser.parse_args()

def main():
    args = get_args()

    if args.id is None or args.imgpath is None or args.outputpath is None:
        print("Problem z argumenti")
        return None

    # image_path = "../images/IMG_8296.jpg"  # Pot do slike
    image_path = args.imgpath
    # output_path = "../images/obraz2.jpg"  # Pot do izhodne slike z obrazom
    output_path = args.outputpath

    # Naloži sliko
    image = cv2.imread(image_path)

    # Pretvori sliko v sivinsko obliko
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ustvari detektor obrazov
    face_cascade = cv2.CascadeClassifier("../haarcascade_frontalface_default.xml")

    # Zazna obraz na sliki
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20,)

    # Preveri, ali je bil obraz zaznan
    if len(faces) > 0:
        # Izreži prvi zaznani obraz
        (x, y, w, h) = faces[0]
        face_image = image[y:y+h, x:x+w]

        # Shrani izrezan obraz na izhodno mesto
        cv2.imwrite(output_path, face_image)
        print("Obraz je bil shranjen.")
    else:
        print("Na sliki ni bilo zaznanega obraza.")

    matrikaSlika = []
    ciljna_velikost = (400, 400)

    user_images = [output_path]  # Seznam poti do slik

    for image_path in user_images:
        image_name = os.path.basename(image_path)  # Dobimo ime slike
        slika = cv2.imread(image_path)

        if slika is not None:
            obrezana_slika = cv2.resize(slika, ciljna_velikost)
            matrikaSlika.append([obrezana_slika, 0])
            print(f"Slika {image_name} je bila uspešno dodana v matriko.")
        else:
            print(f"Napaka pri nalaganju slike {image_name}.")

    značilnice = [slika[0] for slika in matrikaSlika]
    oznake_ucne = [slika[1] for slika in matrikaSlika]
    print(značilnice)

    matrika_značilk_ucne = functions.izlušči_značilnice(matrikaSlika)

    # Branje modela iz datoteke
    # with open("../Model/646d0c1fa7d8e13e080352d7.pkl", "rb") as file:
    try:
        with open(f"../Model/{args.id}.pkl", "rb") as file:
            svm_model = pickle.load(file)
    except:
        print("Napaka pri odpiranju modela!")
        return None

    # Napovedovanje z modelom
    napovedi = svm_model.predict(matrika_značilk_ucne)

    # Primerjava napovedi z oznakami
    for napoved, oznaka in zip(napovedi, oznake_ucne):
        if napoved == oznaka:
            print("Napoved je pravilna.")
        else:
            print("Napoved je napačna.")

if __name__ == "__main__":
    main()
