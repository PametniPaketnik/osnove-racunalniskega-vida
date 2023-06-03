import tensorflow as tf
import os
import cv2


def dodajtag(user_id_folder, person_does_not_exist_folder):
    # Seznam poti do slik uporabnika
    user_images = [os.path.join(user_id_folder, image) for image in os.listdir(user_id_folder)]

    # Seznam poti do slik, kjer oseba ne obstaja
    person_does_not_exist_images = [os.path.join(person_does_not_exist_folder, image) for image in
                                    os.listdir(person_does_not_exist_folder)]

    matrikaUcna = []
    ciljna_velikost = (400, 400)

    for image_path in user_images:
        image_name = os.path.basename(image_path)  # Dobimo ime slike
        slika = cv2.imread(image_path)
        obrezana_slika = cv2.resize(slika, ciljna_velikost)
        matrikaUcna.append([obrezana_slika, 0])

    for imagePDNE_path in person_does_not_exist_images:
        image_name = os.path.basename(imagePDNE_path)  # Dobimo ime slike
        slika = cv2.imread(imagePDNE_path)
        obrezana_slika = cv2.resize(slika, ciljna_velikost)
        matrikaUcna.append([obrezana_slika, 1])

    return matrikaUcna


userId = "3122312321"  # dobimo iz app
users_folder = f"TrainImages/User_{userId}"  # Pot do glavne mape z uporabniki

user_id_folder = os.path.join(users_folder, userId)  # Pot do mape z uporabnikovimi slikami
person_does_not_exist_folder = os.path.join(users_folder, "PersonDoesNotExist")  # Pot do mape s slikami, kjer oseba ne obstaja

matrikaUcna = dodajtag(user_id_folder, person_does_not_exist_folder)

for slika in matrikaUcna:
    print(slika[0], slika[1])
