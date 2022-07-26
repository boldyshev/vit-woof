FROM continuumio/miniconda3


COPY environment.yml /
RUN conda env create -n vit-woof --file environment.yml
RUN conda init bash
RUN conda activate vit-woof
RUN python3 -m pip install -r /requirements.txt

WORKDIR /app
COPY flask_app.py .

EXPOSE 5000
CMD [ "python" , "flask_app.py"]
