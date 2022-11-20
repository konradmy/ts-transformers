PYTHON=python
PIP=pip
export PIP_DEFAULT_TIMEOUT=100


venv:   .FORCE
	pip install --no-cache -r requirements.txt

clean-models:
	rm -R models/checkpoints/* & rm -R models/model_args/* & rm -R models/run_metadata/* & rm -R models_logs/*
.FORCE:
