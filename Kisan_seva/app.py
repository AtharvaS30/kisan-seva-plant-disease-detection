import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =====================
# CONFIG
# =====================
MODEL_PATH = "model/kisan_seva_mobilenetv2.h5"
UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = 224

class_names = os.listdir("PlantVillage_split/train")

# =====================
# LOAD MODEL
# =====================
model = load_model(MODEL_PATH)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================
# HOME PAGE
# =====================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            # Preprocess image
            img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            preds = model.predict(img_array)
            predicted_class = class_names[np.argmax(preds)]

            prediction = predicted_class

    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)
