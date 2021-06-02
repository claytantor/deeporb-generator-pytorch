FROM nvcr.io/nvidia/pytorch:20.06-py3

WORKDIR /usr/src/app

RUN pip3 install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
