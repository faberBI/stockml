FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=8501
RUN pip install --no-cache-dir "streamlit>=1.35" "requests>=2.31" "plotly>=5.20"
COPY app/streamlit_app.py .
EXPOSE 8501
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0"]
