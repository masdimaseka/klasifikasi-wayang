from flask import Flask, request, render_template, redirect, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"  
model = load_model("model_wayang.h5")  

class_descriptions = {
    0: {"name": "Abimanyu", "description": "Abimanyu adalah putra Arjuna yang terkenal akan keberanian dan kecerdasannya dalam strategi perang. Ia gugur dalam perang Bharatayudha saat menghadapi formasi Cakravyuha."},
    1: {"name": "Anoman", "description": "Anoman adalah kera putih sakti dalam cerita Ramayana. Ia dikenal setia kepada Rama dan memiliki kemampuan terbang serta kekuatan luar biasa."},
    2: {"name": "Arjuna", "description": "Arjuna adalah salah satu ksatria Pandawa yang mahir dalam memanah. Ia juga dikenal sebagai sosok yang penuh kebijaksanaan dan memiliki banyak guru."},
    3: {"name": "Bima", "description": "Bima adalah Pandawa kedua yang memiliki kekuatan fisik luar biasa. Ia terkenal dengan senjata gada dan sifatnya yang tegas namun setia kepada saudara-saudaranya."},
    4: {"name": "Cakil", "description": "Cakil adalah tokoh raksasa dalam wayang yang sering menjadi penghalang para ksatria. Ia biasanya digambarkan licik, tetapi selalu kalah dalam setiap pertarungan."},
    5: {"name": "Dusarsana", "description": "Dusarsana adalah saudara Duryudana yang sering membantu rencana jahatnya. Ia terkenal karena sifatnya yang arogan dan penuh ambisi."},
    6: {"name": "Duryudana", "description": "Duryudana adalah pemimpin Kurawa yang ambisius dan licik. Ia sering berkonflik dengan Pandawa demi merebut kekuasaan di Hastinapura."},
    7: {"name": "Gatotkaca", "description": "Gatotkaca adalah anak Bima yang memiliki tubuh besi dan kemampuan terbang. Ia gugur sebagai pahlawan dalam perang Bharatayudha melawan Karna."},
    8: {"name": "Karna", "description": "Karna adalah ksatria hebat yang setia kepada Duryudana meskipun ia sebenarnya saudara Pandawa. Ia dikenal dengan keahlian memanahnya yang luar biasa."},
    9: {"name": "Kresna", "description": "Kresna adalah titisan dewa Wisnu yang menjadi penasihat Pandawa. Ia dikenal bijaksana dan memainkan peran penting dalam kemenangan Pandawa."},
    10: {"name": "Patih Sabrang", "description": "Patih Sabrang adalah salah satu patih dalam kerajaan musuh Pandawa. Ia digambarkan setia kepada rajanya, tetapi sering menjadi korban dalam pertarungan."},
    11: {"name": "Petruk", "description": "Petruk adalah salah satu Punakawan yang dikenal lucu dan bijak. Ia sering memberikan nasihat berharga kepada para ksatria melalui guyonan."},
    12: {"name": "Puntadewa", "description": "Puntadewa adalah pemimpin Pandawa yang dikenal akan kesabaran dan keadilannya. Ia dihormati karena sifatnya yang tidak pernah berbohong."},
    13: {"name": "Semar", "description": "Semar adalah pemimpin Punakawan yang memiliki kebijaksanaan tinggi. Ia sering menjadi penasihat Pandawa dengan petuah yang penuh filosofi."},
    14: {"name": "Sengkuni", "description": "Sengkuni adalah penasihat Kurawa yang licik dan penuh tipu daya. Ia sering memprovokasi Duryudana untuk melawan Pandawa."},
}

os.makedirs("static/images", exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            file = request.files["imagefile"]
            if not file:
                flash("No file selected. Please upload an image.", "danger")
                return redirect(request.url)

            if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                flash("Invalid file type. Please upload an image (PNG, JPG, JPEG).", "danger")
                return redirect(request.url)

            filepath = os.path.join("static/images", file.filename)
            file.save(filepath)

            image = load_img(filepath, target_size=(128, 128))  
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0  

            predictions = model.predict(image)
            confidence = np.max(predictions) * 100  
            predicted_class = np.argmax(predictions)

            if confidence >= 95:
                confidence_category = "Sangat yakin"
            elif confidence >= 90:
                confidence_category = "Cukup yakin"
            elif confidence >= 80:
                confidence_category = "Perlu pengawasan"
            elif confidence >= 70:
                confidence_category = "Tidak yakin"
            else:
                confidence_category = "Kemungkinan besar salah"

            class_info = class_descriptions.get(predicted_class, {"name": "Unknown", "description": "No description available."})

            return render_template(
                "result.html",
                class_name=class_info["name"],
                description=class_info["description"],
                confidence=f"{confidence:.2f}%",
                image_path=f"/static/images/{file.filename}",
                 confidence_category=confidence_category,
            )

        except Exception as e:
            return render_template("result.html", error=f"Error processing image: {str(e)}")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
