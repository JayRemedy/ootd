from flask import Flask, request, render_template
import requests
from PIL import Image
from io import BytesIO
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
import torch

# Import the function from outbound_calls.py
from models import NeuralNetwork


### variables
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# classes
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]






app = Flask(__name__)

# Load your model (ensure this matches your model's setup)
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model_2.pth"))
model.eval()

# Define the transform
transform = Compose([Resize((28, 28)), Grayscale(), ToTensor()])

@app.route('/', methods=['GET'])
def index():
    # Render the HTML form
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the image URL from the form
    image_url = request.form['image_url']

    # Download and process the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = transform(image)
    image = 1 - image  # Invert colors if needed
    image = image.unsqueeze(0).to(device)

    # Classify the image
    with torch.no_grad():
        pred = model(image)
        predicted_class = classes[pred[0].argmax(0)]

    # Return the result
    return f'Predicted Class: {predicted_class}'

if __name__ == '__main__':
    app.run(debug=True)
