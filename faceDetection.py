import cv2
import os

user = "646d0c1fa7d8e13e080352d7" #Dobljen podatek iz app keri user prijavil
# user_str = str(user) # Pripravljeno kak za kak shranjujemo ker ne vem če id al username

image_folder = "images/sabina" # Pot do mape s slikami

output_folder = f"FaceDetection/{user}" # Ustvarite pot do mape z imenom user

os.makedirs(output_folder, exist_ok=True) # Ustvarite mapo z imenom user, če še ne obstaja

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Ustvarite detektor obraza

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        # Ustvarite pot do vhodne slike
        input_image_path = os.path.join(image_folder, filename)

        # Naložite sliko
        image = cv2.imread(input_image_path)

        # Pretvorite sliko v sivinsko sliko za detekcijo obraza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Uporabite detektor obraza na sivi sliki
        faces = facedetect.detectMultiScale(gray, 1.1, 20) # 20 sosedov da je kar se da natancno

        # Če so bili obrazi najdeni, jih shranite v drugo mapo
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Izrežite obraz iz originalne slike
                face = image[y:y + h, x:x + w]

                # Ustvari ime izhodne slike z obrazi
                output_image_path = os.path.join(output_folder, f"face_{filename}")

                # Shrani sliko z obrazi v drugo mapo
                cv2.imwrite(output_image_path, face)

                print(f"Obraz na sliki {filename} je bil zaznan in shranjen v {output_image_path}")
        else:
            print(f"Na sliki {filename} ni bilo najdenih obrazov.")


