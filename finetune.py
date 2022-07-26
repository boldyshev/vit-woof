import json
import argparse

import timm
import torch
from fastai.vision.all import *


def load_labels():
    """Load number labels to breeds dict"""
    with open('breed_labels.json') as json_file:
        breed_labels = json.load(json_file)
    labels = sorted(breed_labels.values())

    return breed_labels, labels


def fastai_dataloader(breed_labels, batch_size):
    """Create ImageWoof dataloader with fastai. """

    # define transformations. Images resized to 224x224
    tfms = [[PILImage.create], [parent_label, breed_labels.__getitem__, Categorize()]]
    item_tfms = [ToTensor(), Resize(224)]
    batch_tfms = [FlipItem(), RandomResizedCrop(224, min_scale=0.35),
                  IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]

    print('Loading dataset...')
    path = untar_data(URLs.IMAGEWOOF)
    items = get_image_files(path)
    split_idx = GrandparentSplitter(valid_name='val')(items)
    dsets = Datasets(items, tfms, splits=split_idx)
    dls = dsets.dataloaders(after_item=item_tfms, after_batch=batch_tfms, bs=batch_size)

    return dls


def load_model(finetune=True, name='vit-woof'):
    print('Loading vision model...', end=' ')
    model = timm.create_model('vit_large_patch16_224',
                              pretrained=finetune,
                              num_classes=10,
                              drop_rate=0.2,
                              attn_drop_rate=0.2)
    if not finetune:
        model_path = f'models/{name}.pt'
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model = torch.nn.Sequential(
            model,
            torch.nn.Softmax(1)
        )

    return model


def main():
    """Finetune vision transformer, pretrained on ImageNet"""

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--bs', default=8, type=int)
    args = parser.parse_args()

    breed_labels, _ = load_labels()
    dataset_downloader = fastai_dataloader(breed_labels, args.bs)
    model = load_model()
    learn = Learner(dataset_downloader,
                    model,
                    loss_func=LabelSmoothingCrossEntropy(),
                    metrics=accuracy)

    # train
    learn.fit_one_cycle(args.epochs, lr_max=args.lr)

    # save
    model_path = f'models/{args.model_name}.pt'
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()

