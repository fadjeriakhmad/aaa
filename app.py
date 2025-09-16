from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model_data = joblib.load('tendangan_model.joblib')
model = model_data['model']
gender_categories = model_data['gender_categories']

@app.route(methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                "PANJANG TUNGKAI (Cm)": float(request.form['panjang_tungkai']),
                "KEKUATAN OTOT TUNGKAI (newton)": float(request.form['kekuatan_otot']),
                "USIA (Tahun)": int(request.form['usia']),
                "JENIS KELAMIN": request.form['jenis_kelamin']
            }

            # Validate input
            if data['JENIS KELAMIN'] not in gender_categories:
                raise ValueError(f"Jenis kelamin harus: {', '.join(gender_categories)}")

            # Predict
            input_df = pd.DataFrame([data])
            prediction = round(float(model.predict(input_df)[0]), 1)

        except ValueError as e:
            prediction = f"Input tidak valid: {str(e)}"
        except Exception as e:
            prediction = f"Terjadi error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

