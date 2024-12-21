mkdir ai_toon_webapp
cd ai_toon_webapp
python3 -m venv venv
source venv/bin/activate
pip install flask


from flask import Flask, request, jsonify
from PIL import Image
import torch
import io

app = Flask(__name__)

# Load pre-trained model
model = torch.hub.load('znxlwm/pytorch-CartoonGAN', 'CartoonGAN')

@app.route('/convert', methods=['POST'])
def convert_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(file.stream)

    # Transform and process image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        cartoon_img = model(img_t)

    cartoon_img = cartoon_img.squeeze().permute(1, 2, 0).numpy()
    cartoon_img = (cartoon_img * 255).astype(np.uint8)
    cartoon_img = Image.fromarray(cartoon_img)

    # Save or return the cartoon image
    img_byte_arr = io.BytesIO()
    cartoon_img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    return jsonify({'cartoon_image': img_byte_arr})

if __name__ == '__main__':
    app.run(debug=True)
