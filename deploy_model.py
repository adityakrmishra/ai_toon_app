from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Load the trained generator model
generator = tf.keras.models.load_model('path_to_saved_generator_model')

app = Flask(__name__)

@app.route('/toonify', methods=['POST'])
def toonify():
    file = request.files['image']
    image = Image.open(file.stream)
    image = image.resize((256, 256))
    image = np.array(image) / 127.5 - 1.0
    image = np.expand_dims(image, axis=0)

    cartoon_image = generator.predict(image)
    cartoon_image = (cartoon_image[0] + 1.0) * 127.5
    cartoon_image = cartoon_image.astype(np.uint8)

    img = Image.fromarray(cartoon_image)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return jsonify({'cartoon_image': img_byte_arr})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
