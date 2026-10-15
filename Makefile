export PYTHONPATH := src

.PHONY: install evaluate app api test lint docker

install:    ## install dependencies
	pip install -r requirements-dev.txt
evaluate:   ## generate documents, run the agent, score it -> outputs/report.md
	python -m docagent.pipeline
app:        ## http://localhost:8501
	streamlit run app.py
api:        ## http://localhost:8000/docs
	uvicorn api:app --reload
test:
	pytest -q
lint:
	ruff check .
docker:
	docker build -t uae-document-intelligence-agent . && docker run -p 8501:8501 uae-document-intelligence-agent
