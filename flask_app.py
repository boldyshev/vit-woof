import re
import base64
from io import BytesIO
import argparse

import torch
import torchvision.transforms as T
from PIL import Image
from fastai.vision.all import imagenet_stats

from flask import Flask, render_template, request, jsonify

from finetune import load_labels, load_model

app = Flask(__name__)


def classify(img_data, model, labels):
    """Transform image to tensor and get prediction"""

    img = Image.open(img_data)

    # transform
    transform = T.Compose([T.Resize((224, 224)),
                           T.ToTensor(),
                           T.Normalize(*imagenet_stats)])

    resized_img = transform(img).unsqueeze(0)

    # predict
    preds = model.forward(resized_img).squeeze(0)
    idx = torch.argmax(preds)
    breed, prob = labels[idx], preds[idx]
    result = f'{breed}: {100 * prob:.2f}%'

    return result


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        # get the image from post request
        image_data = re.sub('^data:image/.+;base64,', '', request.json)
        image_data = BytesIO(base64.b64decode(image_data))

        result = classify(image_data, model, labels)

        # serialize the result, you can add additional fields
        return jsonify(result=result)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model_name', default='vit-woof.pt', help='name of the model to load from /models')
    args = parser.parse_args()
    _, labels = load_labels()
    model = load_model(finetune=False, name=args.model_name)
    app.run(host='0.0.0.0')
