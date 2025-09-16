from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
interpreter = tf.lite.Interpreter(model_path="sign.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/', methods=['GET', 'POST'])
def detect():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(file.stream).resize((300, 300))  # Ukuran tergantung model
        input_data = np.expand_dims(np.array(image, dtype=np.uint8), axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        result = output_data.tolist()  # Bisa diolah lebih lanjut

    return render_template('index.html', result=result)