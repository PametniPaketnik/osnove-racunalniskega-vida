import cv2
import numpy as np
from skimage.feature import hog
import os


def lbp(sivinska_slika):
    # Določanje velikosti slike
    visina, sirina = sivinska_slika.shape

    # Izračun LBP slikovne značilnice
    lbp_slika = np.zeros((visina, sirina), dtype=np.uint8)
    for i in range(1, visina-1):
        for j in range(1, sirina-1):
            center = sivinska_slika[i, j]  # središčna vrednost
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
    histogram, _ = np.histogram(lbp_slika.ravel(), bins=np.arange(256 + 1), range=(0, 256))  # 1d, meje vrednosti, območje vrednosti
    histogram = histogram.astype("float")  # decimal
    # Normalizacija histograma
    histogram /= (histogram.sum() + 1e-7)  # vsota vrednosti -> vsako število v h deli s to vsoto

    return histogram


def izracunaj_hog(slika, vel_celice, vel_blok, razdelki):
    # Izračunaj značilnice HOG
    hog_značilnice, _ = hog(slika, orientations=razdelki, pixels_per_cell=(vel_celice, vel_celice),
                            cells_per_block=(vel_blok, vel_blok), block_norm='L2-Hys', visualize=True)


def izlušči_značilnice(matrika):
    značilnice = []
    vel_celice = 4
    vel_blok = 2  # Koliko celic je v enem bloku
    razdelki = 15

    for slika in matrika:
        slikadobljena, oznaka = slika
        slika_siva = cv2.cvtColor(slikadobljena, cv2.COLOR_BGR2GRAY) # nastavi na gray

        # Izlušči značilnice LBP
        lbp_značilnice = lbp(slika_siva)

        # Izlušči značilnice HOG
        hog_značilnice = izracunaj_hog(slika_siva, vel_celice, vel_blok, razdelki)

        # Združi značilnice LBP in HOG
        značilnice_slike = np.concatenate((lbp_značilnice, hog_značilnice))

        značilnice.append(značilnice_slike)
        # print(značilnice)

    return np.array(značilnice)


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


def zaznajObraz(image_path, output_path):
    # Naloži sliko
    image = cv2.imread(image_path)

    # Pretvori sliko v sivinsko obliko
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ustvari detektor obrazov
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Zazna obraz na sliki
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, )

    # Preveri, ali je bil obraz zaznan
    if len(faces) > 0:
        # Izreži prvi zaznani obraz
        (x, y, w, h) = faces[0]
        face_image = image[y:y + h, x:x + w]

        # Shrani izrezan obraz na izhodno mesto
        cv2.imwrite(output_path, face_image)
        print("Obraz je bil shranjen.")
    else:
        print("Na sliki ni bilo zaznanega obraza.")
