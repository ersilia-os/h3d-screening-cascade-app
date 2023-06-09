FROM python:3.10.7-bullseye

WORKDIR .

COPY . .

RUN python -m pip install --upgrade pip
RUN python -m pip install streamlit
RUN git clone https://github.com/ersilia-os/compound-embedding-lite
RUN python -m pip install -e compound-embedding-lite/.
RUN python -m pip install flaml

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py"]