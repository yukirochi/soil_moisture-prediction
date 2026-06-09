FROM python:3.11

WORKDIR /soil_moisture_prediction

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "model/model.py"]

