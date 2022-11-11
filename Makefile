PYTHON=python
PIP=pip
export PIP_DEFAULT_TIMEOUT=100


venv:   .FORCE
	pip install --no-cache -r requirements.txt

.FORCE:
