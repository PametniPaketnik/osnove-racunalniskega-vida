# Face detection

The application enables the use of biometric data recognition for the purpose of registration or user verification to your mobile application.

# Requirements

```bash
pip install opencv-python numpy tensorflow argparse scikit-image scikit-learn shutil
```

# Installation

Install from source
```bash
git clone https://github.com/PametniPaketnik/osnove-racunalniskega-vida
```

# Run

First you need to create a model
```bash
python src/model.py --id "646d0c1fa7d8e13e080352d7" --usersfolder "../images/TrainImages"
```

To check if thats the user
```bash
python src/login.py --id "646d0c1fa7d8e13e080352d7" --imgpath "../images/IMG_8296.jpg" --outputpath "../images/obraz2.jpg"
```
