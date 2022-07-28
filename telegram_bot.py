import os
import argparse
import telebot

import timm
from flask_app import output
from finetune import load_labels, load_model

# get telegram API token from console
parser = argparse.ArgumentParser()
parser.add_argument('token', help='telegram bot token to access the HTTP API')
parser.add_argument('-n', '--model_name', default='vit-woof.pt', help='name of the model to load from /models')
args = parser.parse_args()

welcome_text = '''
Hi there!\nI classify doggies of the following breeds:

Shih-Tzu,
Rhodesian ridgeback,
Beagle,
English foxhound,
Border terrier,
Australian terrier,
Golden retriever,
Old English sheepdog,
Samoyed,
Dingo.

Send me a picture of one of these.
'''
_, labels = load_labels()


@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(message.chat.id, welcome_text)


@bot.message_handler(content_types=['text'])
def need_photo(message):
    bot.send_message(message.chat.id, 'I need a photo')


@bot.message_handler(content_types=['photo'])
def guess_breed(message):

    # get image info
    bot.send_message(message.chat.id, 'Let me think...')
    img_info = bot.get_file(message.photo[-1].file_id)

    # temporarily download image
    downloaded_img = bot.download_file(img_info.file_path)
    img_path = f'images/{message.photo[-1].file_id}'

    # open image locally and classify
    with open(img_path, 'wb') as new_file:
        new_file.write(downloaded_img)

    result = output(img_path, model_dog, model_breed)

    # show result
    bot.send_message(message.chat.id, result)

    # delete image
    os.remove(img_path)


if __name__ == "__main__":
    model_dog = timm.create_model('vit_large_patch16_224', pretrained=True)
    model_breed = load_model(finetune=False, name=args.model_name)

    print('Bot ready')
    bot.polling(none_stop=True)
