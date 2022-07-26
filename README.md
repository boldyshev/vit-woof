# Тестовое задание по CV:

- Реализовать пайплайн CV классификации изображений с помощью ***Pytorch***.
- Создать сервис на Flask, который может классифицировать картинки собак
- Обернуть его в Docker *
- Создать телеграм бота для классификации собак *

Датасет [ImageWoof](https://github.com/fastai/imagenette#imagewoof)

## Архитектура: Vision Transformer

Причины:
- На июль 2022 лидирует в классификации [ImageNet](https://www.google.com/search?q=imagenet+leaderboard&oq=imagenet&aqs=chrome.1.69i59l2j0i512l3j69i60l3.1709j0j7&sourceid=chrome&ie=UTF-8)
- Дает лучший результат, чем архитектуры из 
[leaderboard ImageWoof](https://github.com/fastai/imagenette#imagewoof-leaderboard) на 2020-21
  (cм. сравнение в [pipeline.ipynb](pipeline.ipynb) )

## Полученные метрики

![vit_training.png](images/vit_training.png) 

Validation accuracy = 0.93  
https://forums.fast.ai/t/training-loss-and-training-set-accuracy/14302/7

Предобученная на ImageNet модель ```vit_large_patch16_224``` дообучилась до этих значений за одну эпоху.


## Разведочный анализ
???

## Анализ ошибок

Confusion matrix

![vit_conf.png](images/vit_conf.png) 

Наиболее частые ошибки
```
[('English foxhound', 'Beagle', 41), 
('Beagle', 'English foxhound', 20)]
```



