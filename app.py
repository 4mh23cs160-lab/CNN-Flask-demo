import os
import io
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# Load pre-trained ResNet18 model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

# Transform for specific model
preprocess = weights.transforms()

@app.route("/", methods=["GET"])
def hello_world():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return redirect("/")
    
    file = request.files['file']
    if not file:
        return redirect("/")

    try:
        # Read and preprocess image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        batch = preprocess(img).unsqueeze(0)

        # Inference
        with torch.no_grad():
            prediction = model(batch)
            probabilities = F.softmax(prediction, dim=1)
        
        # Get top prediction
        class_id = prediction.argmax().item()
        score = probabilities[0][class_id].item()
        category_name = weights.meta["categories"][class_id]
        
        return render_template('index.html', result=f"{category_name} ({score:.2%})")

    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)