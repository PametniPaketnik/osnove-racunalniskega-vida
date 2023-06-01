import cv2
import os


image_folder = "PersonDoesNotExist" # Pot do mape s slikami

output_folder = f"FaceDetection/PersonDoesNotExist" # Ustvarite pot do mape z imenom PersonDoesNotExist

os.makedirs(output_folder, exist_ok=True) # Ustvarite mapo z imenom PersonDoesNotExist, če še ne obstaja

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Detektor obraza

# Preberite vse datoteke s končnico .jfif iz mape s slikami
for filename in os.listdir(image_folder):
    if filename.endswith(".jfif"):

        # Ustvarite pot do vhodne slike
        input_image_path = os.path.join(image_folder, filename)

        # Naložite sliko
        image = cv2.imread(input_image_path)

        # Pretvorite sliko v sivinsko sliko za detekcijo obraza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Uporabite detektor obraza na sivi sliki
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20)


        # Če so bili obrazi najdeni, jih shranite v drugo mapo
        if len(faces) > 0:
            for (x, y, w, h) in faces:

                # Izrežite obraz iz originalne slike
                face = image[y:y + h, x:x + w]

                # Ustvarite ime izhodne slike z obrazi
                filename = filename.split(".jfif")[0] # odstrani .jfif
                output_image_path = os.path.join(output_folder, f"face_{filename}.jpg")

                # Shranite sliko z obrazi v drugo mapo
                cv2.imwrite(output_image_path, face)

                print(f"Obraz na sliki {filename} je bil zaznan in shranjen v {output_image_path}")
        else:
            print(f"Na sliki {filename} ni bilo najdenih obrazov.")


