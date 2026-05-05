from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
import webbrowser
import threading

app = Flask(__name__)


model = load_model("model.h5")


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


class_names = [ "Healthy" , "disease1", "disease2"]

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

      
        img = Image.open(path)
        img = img.resize((128, 128))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

      
        prediction = model.predict(img)

        confidence = float(np.max(prediction) * 100)
        result = class_names[np.argmax(prediction)]

        return render_template(
            "index.html",
            prediction=result,
            confidence=round(confidence, 2),
            image=file.filename
        )

    return "No file uploaded"



def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False)