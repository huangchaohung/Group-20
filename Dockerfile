FROM python:3.12

WORKDIR /Group-20/src

RUN python -m venv venv

ENV PATH="/server/.venv/bin:$PATH"

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "main.py"]
