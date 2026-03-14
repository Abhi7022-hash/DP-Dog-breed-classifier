# Dog Breed Classifier 🐶

## Overview

This project is a deep learning based **Dog Breed Classifier** that predicts the breed of a dog from an input image.
The model uses **Transfer Learning with InceptionV3** to classify dog breeds accurately.

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV / PIL

---

## Model

We used **InceptionV3**, a pre-trained convolutional neural network trained on the ImageNet dataset.
The model is fine-tuned to classify different dog breeds from images.

---

## Project Structure

```
dog-breed-classifier
│
├── src
│   ├── train.py
│   ├── predict.py
│   └── model.py
│
├── docs
├── README.md
├── requirements.txt
├── architecture.png
├── demo_video_link.txt
└── setup_instructions.md
```

---

## How to Run

Clone the repository

```
git clone https://github.com/Abhi7022-hash/dog-breed-classifier.git
```

Go to project folder

```
cd dog-breed-classifier
```

Install dependencies

```
pip install -r requirements.txt
```

Run the project

```
python src/train.py
```

---

## Output

The model takes a dog image as input and predicts the **dog breed** as output.

---

## Author

Abhishek Dindavar
