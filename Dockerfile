FROM nvcr.io/nvidia/pytorch:20.06-py3

WORKDIR /usr/src/app

RUN apt-get update -y
RUN apt-get install -y fluidsynth-dssi fluidsynth

RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8001"]
