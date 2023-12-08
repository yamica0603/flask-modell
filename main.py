from flask import Flask, render_template, request
from model_p import predict_result, preprocess_img

app = Flask(__name__, template_folder="templates")


# home route
@app.route("/")
def main():
    return render_template("index copy.html")


# predict route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.method == "POST":
            if "file" not in request.files:
                error = "No file part"
                return render_template("results.html", err=error)

            file = request.files["file"]
            # logging
            print(f"Received file: {file.filename}")

            img_stream = file.stream
            img = preprocess_img(img_stream)

            pred = predict_result(img)

            return render_template("results.html", prediction=str(pred))

    except Exception as e:
        error = f"Error: {str(e)}"
        return render_template("results.html", err=error)


if __name__ == "__main__":
    app.run()
