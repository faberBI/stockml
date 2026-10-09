.PHONY: install train api app test lint up
install: ; pip install -e ".[api,app,dev]"
train:   ; stockml-train
api:     ; uvicorn stockml.api.main:app --reload
app:     ; API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
test:    ; pytest -q
lint:    ; ruff check src app tests
up:      ; docker compose up --build
