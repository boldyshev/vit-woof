import os
import telebot

from flask_app import classify
from finetune import load_labels, load_model

API_TOKEN = '5575577930:AAFMjSEeVIQz0rwMywi92Y_tzWcKc7xXctk'
bot = telebot.TeleBot(API_TOKEN)

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


@bot.message_handler(commands=['start'])
def welcome(message):
    bot.send_message(message.chat.id, welcome_text)


@bot.message_handler(content_types=['text'])
def need_photo(message):
    bot.send_message(message.chat.id, 'I need a photo')


@bot.message_handler(content_types=['photo'])
def guess_breed(message):
    bot.send_message(message.chat.id, 'Let me think...')
    img_info = bot.get_file(message.photo[-1].file_id)
    downloaded_img = bot.download_file(img_info.file_path)
    img_path = f'images/{message.photo[-1].file_id}'

    with open(img_path, 'wb') as new_file:
        new_file.write(downloaded_img)

    pred = classify(img_path, model, labels)

    bot.send_message(message.chat.id, pred)
    os.remove(img_path)


if __name__ == "__main__":
    _, labels = load_labels()
    model = load_model(finetune=False, name='vit-woof')
    print('Bot ready.')
    bot.polling(none_stop=True)
