FROM continuumio/miniconda3

COPY environment.yml /
RUN conda env create -n vit-woof --file environment.yml
RUN conda activate vit-woof
RUN python3 -m pip install -r /requirements.txt

COPY . /flask_app
WORKDIR /flask_app

EXPOSE 5000
CMD [ "python" , "flask_app.py"]
