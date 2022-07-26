FROM continuumio/miniconda3

COPY environment.yml /
RUN conda env create -n vit-woof --file environment.yml
SHELL ["conda", "run", "-n", "vit-woof", "/bin/bash", "-c"]

COPY . /vit-woof
WORKDIR /vit-woof

#COPY flask_app.py .
#COPY finetune.py .
#COPY breed_labels.json .
#COPY models/vit-woof.pt .

EXPOSE 5000
CMD [ "conda", "run", "-n", "vit-woof", "python" , "flask_app.py"]
