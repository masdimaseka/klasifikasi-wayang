from flask import Flask, request, render_template, redirect, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

from class_descriptions import class_descriptions  # Import deskripsi karakter

# Inisialisasi Flask
app = Flask(__name__)
app.secret_key = "supersecretkey"  
model = load_model("model_wayang.h5")  

# Pastikan folder untuk menyimpan gambar tersedia
UPLOAD_FOLDER = "static/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_confidence_category(confidence):
    """Mengembalikan kategori confidence tanpa loop."""
    if confidence >= 95:
        return "Sangat yakin"
    elif confidence >= 90:
        return "Cukup yakin"
    elif confidence >= 80:
        return "Perlu pengawasan"
    elif confidence >= 70:
        return "Tidak yakin"
    else:
        return "Kemungkinan besar salah"

def process_image(image_path):
    """Memproses gambar dan mengembalikan prediksi kelas serta confidence."""
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  

    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("imagefile")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            predicted_class, confidence = process_image(filepath)
            confidence_category = get_confidence_category(confidence)

            class_info = class_descriptions.get(
                predicted_class, 
                {"name": "Unknown", "description": "No description available."}
            )

            return render_template(
                "result.html",
                class_name=class_info["name"],
                description=class_info["description"],
                confidence=f"{confidence:.2f}%",
                image_path=f"/{filepath}",
                confidence_category=confidence_category,
            )

        except Exception as e:
            flash("Error processing image. Please try again.", "danger")
            return redirect(request.url)

    return render_template("index.html")

@app.route("/list")
def list_wayang():
    return render_template("list.html", wayang_list=class_descriptions)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
