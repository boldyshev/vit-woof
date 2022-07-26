FROM continuumio/miniconda3


COPY environment.yml /
RUN conda env create -n vit-woof --file environment.yml
SHELL ["conda", "run", "-n", "vit-woof", "/bin/bash", "-c"]

WORKDIR /app
COPY flask_app.py .

EXPOSE 5000
CMD [ "conda", "run", "-n", "vit-woof", "python" , "flask_app.py"]
