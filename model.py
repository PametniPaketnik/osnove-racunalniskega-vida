import os

userId = "3122312321"  # dobimo iz app
users_folder = f"TrainImages/User_{userId}"  # Pot do glavne mape z uporabniki

user_id_folder = os.path.join(users_folder, userId)  # Pot do mape z uporabnikovimi slikami
person_does_not_exist_folder = os.path.join(users_folder, "PersonDoesNotExist")  # Pot do mape s slikami, kjer oseba ne obstaja

user_images = []  # Seznam poti do slik uporabnika
person_does_not_exist_images = []  # Seznam poti do slik, kjer oseba ne obstaja

# Pridobitev poti do slik uporabnika
for filename in os.listdir(user_id_folder):
    if filename.endswith(".jpg"):
        user_images.append(os.path.join(user_id_folder, filename))

# Pridobitev poti do slik, kjer oseba ne obstaja
for filename in os.listdir(person_does_not_exist_folder):
    if filename.endswith(".jpg"):
        person_does_not_exist_images.append(os.path.join(person_does_not_exist_folder, filename))

# Izpis poti do slik uporabnika
print("Slike uporabnika:")
for image_path in user_images:
    print(image_path)

# Izpis poti do slik, kjer oseba ne obstaja
print("Slike, kjer oseba ne obstaja:")
for image_path in person_does_not_exist_images:
    print(image_path)
