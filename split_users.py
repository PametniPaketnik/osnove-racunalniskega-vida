import os
import shutil

# Ustvarite ime in pot do nove mape
userId = "646d0c1fa7d8e13e080352d7" # dobimo iz app
newUser_folder = "User_" + userId #Nova mapa

# Poti do map kjer imamo ze narejene obraze
personDoesNotExist = "FaceDetection/PersonDoesNotExist"
userId_folder = f"FaceDetection/{userId}"

existing_folder = "TrainImages"  # Pot do že obstoječe mape "TrainImages"
# Ustvarite pot do nove mape znotraj že obstoječe mape
output_folder = os.path.join(existing_folder, newUser_folder)

# Ustvarite novo mapo, če še ne obstaja
os.makedirs(output_folder, exist_ok=True)

# Poti do dveh map, ki ju želite kopirati
source_folder1 = personDoesNotExist
source_folder2 = userId_folder

# Kopirajte vsebino prve mape v novo mapo
shutil.copytree(source_folder1, os.path.join(output_folder, os.path.basename(source_folder1)))

# Kopirajte vsebino druge mape v novo mapo
shutil.copytree(source_folder2, os.path.join(output_folder, os.path.basename(source_folder2)))

print("Mape so bile uspešno kopirane v novo mapo.")