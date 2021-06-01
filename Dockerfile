FROM nvcr.io/nvidia/pytorch:20.06-py3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
