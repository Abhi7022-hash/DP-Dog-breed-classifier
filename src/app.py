from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import base64

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = tf.keras.models.load_model("final_inception_model_299.h5")

# Load class labels
with open("class_indices.json") as f:
    class_indices = json.load(f)

labels = {v:k for k,v in class_indices.items()}

# Load breed info
with open("breed_info.json") as f:
    breed_info = json.load(f)

IMG_SIZE = 299

def prepare_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/", methods=["GET","POST"])
def index():
    prediction = None
    description = None
    filename = None

    if request.method == "POST":

        # 🔹 CAMERA IMAGE HANDLING
        if "cam_image" in request.form and request.form["cam_image"]:
            data = request.form["cam_image"].split(",")[1]
            image_bytes = base64.b64decode(data)

            filename = "camera.png"
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            with open(filepath, "wb") as f:
                f.write(image_bytes)

        # 🔹 FILE UPLOAD HANDLING
        elif "file" in request.files:
            file = request.files["file"]

            if file.filename != "":
                filename = file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
            else:
                filepath = None

        else:
            filepath = None

        # 🔹 PREDICTION
        if filepath:
            img = Image.open(filepath).convert("RGB")
            processed = prepare_image(img)

            pred = model.predict(processed)
            class_id = np.argmax(pred)
            confidence = np.max(pred)

            breed = labels[class_id]
            prediction = f"{breed} ({confidence*100:.2f}%)"

            info = breed_info.get(breed)

            if info:
                description = info
            else:
                description = {
                    "features": "No information available.",
                    "found": "No information available.",
                    "nature": "No information available."
                }

    return render_template(
        "index.html",
        prediction=prediction,
        description=description,
        filename=filename
    )

if __name__ == "__main__":
    app.run(debug=True)
