import tensorflow as tf
import os
import cv2
import numpy as np

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

def lbp(sivinska_slika):
    # Določanje velikosti slike
    visina, sirina = sivinska_slika.shape

    # Izračun LBP slikovne značilnice
    lbp_slika = np.zeros((visina, sirina), dtype=np.uint8)
    for i in range(1, visina-1):
        for j in range(1, sirina-1):
            center = sivinska_slika[i, j] # središčna vrednost
            # vrednosti sosedov centra se določijo levo, desno, gor, dolj
            vrednosti_sosedov = [
                sivinska_slika[i-1, j-1], sivinska_slika[i-1, j], sivinska_slika[i-1, j+1],
                sivinska_slika[i, j-1],                           sivinska_slika[i, j+1],
                sivinska_slika[i+1, j-1], sivinska_slika[i+1, j], sivinska_slika[i+1, j+1]
            ]
            # Vrednost sosedov pretvori v binarne vrednosti glede na center (vrednost soseda večja ali enaka == 1 else 0
            vrednosti_sosedov_bin = [int(v >= center) for v in vrednosti_sosedov]
            # binarna vrednost pomnozi z  2^8 ter nato seštejejo
            lbp_vrednost = sum([vrednosti_sosedov_bin[k] * (2**k) for k in range(8)])
            lbp_slika[i, j] = lbp_vrednost  # Izračunana lbp vrednost shrani v matriko na ustrezni položaj

    # Izračun histograma LBP slikovne značilnice
    histogram, _ = np.histogram(lbp_slika.ravel(), bins=np.arange(256 + 1), range=(0, 256)) # 1d, meje vrednosti, območje vrednosti
    histogram = histogram.astype("float")  # decimal
    # Normalizacija histograma
    histogram /= (histogram.sum() + 1e-7)  # vsota vrednosti -> vsako število v h deli s to vsoto

    return histogram


userId = "3122312321"  # dobimo iz app
users_folder = f"TrainImages/User_{userId}"  # Pot do glavne mape z uporabniki

user_id_folder = os.path.join(users_folder, userId)  # Pot do mape z uporabnikovimi slikami
person_does_not_exist_folder = os.path.join(users_folder, "PersonDoesNotExist")  # Pot do mape s slikami, kjer oseba ne obstaja

matrikaUcna = dodajtag(user_id_folder, person_does_not_exist_folder)
oznake_ucne = [slika[1] for slika in matrikaUcna]
print(oznake_ucne)

'''
for slika in matrikaUcna:
    print(slika[0], slika[1])
'''