.PHONY: help venv install install-dev install-all clean notebook app docker docker-run lint format

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PYTHON       ?= python3.12
PIP          ?= $(PYTHON) -m pip
PORT_APP     ?= 8501
DOCKER_IMAGE ?= laliga-hypermotion-predictor
DOCKER_TAG   ?= latest

# ---------------------------------------------------------------------------
# Ayuda
# ---------------------------------------------------------------------------
help:  ## Listado de comandos disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------
venv:  ## Crea entorno virtual .venv con Python 3.12
	$(PYTHON) -m venv .venv
	@echo "  -> Activa con: source .venv/bin/activate"

install:  ## Instala dependencias core (sin ml/app/dev)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .

install-dev:  ## Instala core + ml + app + dev (JupyterLab incluido)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,ml,app]"

install-all:  ## Instala TODO incluido PyTorch (pesado)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,ml,app,nn]"

clean:  ## Borra artefactos temporales (no toca datos ni modelos)
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Notebooks (flujo principal del TFM)
# ---------------------------------------------------------------------------
notebook:  ## Abre JupyterLab para trabajar con los notebooks 0X_*.ipynb
	jupyter lab

# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------
lint:  ## Lint con ruff (src + notebooks)
	ruff check src notebooks

format:  ## Formatea con ruff
	ruff format src notebooks
	ruff check --fix src notebooks

# ---------------------------------------------------------------------------
# App Streamlit + Docker (despliegue Azure Container Apps)
# ---------------------------------------------------------------------------
app:  ## Lanza la app Streamlit en local (http://localhost:8501)
	streamlit run app/streamlit_app.py --server.port=$(PORT_APP)

docker:  ## Construye la imagen Docker de la app
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) -f app/Dockerfile .

docker-run:  ## Ejecuta la app en Docker
	docker run --rm -p $(PORT_APP):8501 $(DOCKER_IMAGE):$(DOCKER_TAG)
