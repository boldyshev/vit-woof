import re
import base64
from io import BytesIO
import argparse


import torch
import torchvision.transforms as T
from PIL import Image
from fastai.vision.all import imagenet_stats
import timm

from flask import Flask, render_template, request, jsonify

from finetune import load_labels, load_model

app = Flask(__name__)
_, labels = load_labels()


def predict(img_data, model):
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

    return idx


def output(img_path, model_dog, model_breed):
    """Predicts breed if inout is a dof image, otherwise returns 'Not a dog'"""

    # In ImageNet dogs have labels from 151 to 268
    dog_image = predict(img_path, model_dog) in range(151, 269)
    if dog_image:
        idx = predict(img_path, model_breed)
        result = labels[idx]
    else:
        result = 'Not a dog'

    return result


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':

        # get the image from post request
        image_data = re.sub('^data:image/.+;base64,', '', request.json)
        image_data = BytesIO(base64.b64decode(image_data))

        # result = output(image_data, model_dog, model_breed)
        result = 'lala'
        # serialize the result, you can add additional fields
        return jsonify(result=result)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model_name', default='vit-woof.pt', help='name of the model to load from /models')
    args = parser.parse_args()
    # model_dog = timm.create_model('vit_large_patch16_224', pretrained=True)
    model_breed = load_model(finetune=False, name=args.model_name)
    app.run(host='0.0.0.0')
