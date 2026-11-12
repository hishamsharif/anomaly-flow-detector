.PHONY: help setup resetup start stop logs verify test lint migrate export-spec clean-models api ui

COMPOSE        = docker compose
DL_ENV         = /home/hhaniffa/miniconda/envs/afd_v2/bin
DOCKER_RUNNING = docker compose ps --services --filter status=running 2>/dev/null | grep -q api

# Parse positional component from MAKECMDGOALS at parse time (before any target definition)
_COMP := $(or $(filter api ui,$(MAKECMDGOALS)),both)

# `make <cmd> api` / `make <cmd> ui` — no-op targets; must come AFTER _COMP is set
api ui:
	@:

.DEFAULT_GOAL := help

# ── Help (default target) ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Anomaly Flow Detector — available commands"
	@echo ""
	@echo "  Setup (run once)"
	@echo "    make setup            Create afd_v2 env + deps (GPU if CUDA found) + dataset + migrate"
	@echo "    make resetup          Wipe env + DB + models, then run setup from scratch"
	@echo ""
	@echo "  Stack"
	@echo "    make start            Start full stack  (Docker if available, local guide otherwise)"
	@echo "    make start api        Start API locally on :8000  (hot-reload, logs → logs/api.log)"
	@echo "    make start ui         Start UI dev server on :5173  (logs → logs/ui.log)"
	@echo "    make stop             Stop the Docker stack"
	@echo ""
	@echo "  Logs  (auto-detects Docker or local log files)"
	@echo "    make logs             Follow all logs"
	@echo "    make logs api         Follow API logs only"
	@echo "    make logs ui          Follow UI logs only"
	@echo ""
	@echo "  Verify"
	@echo "    make verify           Check both API and UI are reachable"
	@echo "    make verify api       Check API /health endpoint"
	@echo "    make verify ui        Check UI is serving on :5173"
	@echo ""
	@echo "  Development"
	@echo "    make test             Run pytest suite"
	@echo "    make lint             Ruff check + format check"
	@echo "    make migrate          Apply Alembic migrations"
	@echo "    make export-spec      Write docs/openapi.json"
	@echo "    make clean-models     Delete versioned model dirs under models/"
	@echo ""

# ── Setup (run once) ──────────────────────────────────────────────────────────
setup:
	@echo ""
	@if conda env list | grep -q "^afd_v2 "; then \
		echo "  afd_v2 env already exists."; \
		printf "  Did you mean 'make resetup' for a full clean slate? [y/N] "; \
		read confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			echo "  → Run: make resetup"; exit 0; \
		fi; \
		echo "  Continuing — reinstalling dependencies into existing env …"; \
		echo ""; \
	else \
		echo "==> Creating conda environment (afd_v2, Python 3.12)"; \
		conda create -n afd_v2 python=3.12 -y; \
	fi
	@echo "==> Installing dependencies"
	conda run -n afd_v2 pip install -r requirements.txt
	@if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
		echo "  CUDA detected — installing cu121 torch"; \
		conda run -n afd_v2 pip install torch==2.5.1 \
			--index-url https://download.pytorch.org/whl/cu121; \
	else \
		echo "  No CUDA detected — using CPU torch (re-run setup after enabling GPU)"; \
	fi
	@echo ""
	@echo "==> Dataset"
	@CSV_COUNT=$$(find -L data/CIC-IDS2017 -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l); \
	if [ "$$CSV_COUNT" -gt 0 ]; then \
		echo "  CIC-IDS2017 already present ($$CSV_COUNT CSV files) — skipping."; \
	else \
		mkdir -p data/CIC-IDS2017; \
		echo "  Downloading CIC-IDS2017 (~225 MB) …"; \
		conda run -n afd_v2 gdown --id 10UG7eEAUI8kV4_CMFsrK-5nVz4Bm0Ey3 -O data/CIC-IDS2017.zip; \
		unzip -q data/CIC-IDS2017.zip -d data/CIC-IDS2017/; \
		rm data/CIC-IDS2017.zip; \
		echo "  Done — $$(find -L data/CIC-IDS2017 -maxdepth 1 -name '*.csv' | wc -l) CSV files ready."; \
	fi
	@echo ""
	@echo "==> Applying DB migrations"
	@mkdir -p data; [ -f data/afd.db ] || touch data/afd.db
	$(MAKE) migrate
	@echo ""
	@echo "Setup complete. Next: make start"

# ── Resetup (clean-slate rebuild) ─────────────────────────────────────────────
resetup:
	@echo ""
	@echo "  WARNING: resetup will permanently delete:"
	@echo "    • afd_v2 conda env  (all Python packages)"
	@echo "    • afd.db            (all alerts, verdicts, model history)"
	@echo "    • models/           (all trained model artifacts)"
	@echo "    • logs/             (all log files)"
	@echo ""
	@printf "  Proceed? This cannot be undone. [y/N] "; \
	read confirm; \
	if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then \
		echo "Aborted."; exit 0; \
	fi; \
	echo ""; \
	echo "==> Stopping stack…"; \
	$(COMPOSE) down 2>/dev/null; true; \
	echo "==> Removing conda env…"; \
	conda env remove -n afd_v2 -y 2>/dev/null; true; \
	echo "==> Clearing database…"; \
	rm -f data/afd.db; \
	mkdir -p data; \
	touch data/afd.db; \
	echo "==> Clearing model artifacts…"; \
	$(MAKE) clean-models; \
	echo "==> Clearing logs…"; \
	rm -f logs/*.log 2>/dev/null; true; \
	echo ""; \
	echo "Clean slate — running setup…"; \
	echo ""
	$(MAKE) setup

# ── Stack ──────────────────────────────────────────────────────────────────────
start:
	@if [ "$(_COMP)" = "api" ]; then \
		lsof -ti:8000 2>/dev/null | xargs -r kill -9 2>/dev/null; true; \
		mkdir -p logs; \
		PYTHONPATH=. $(DL_ENV)/alembic upgrade head && \
		PYTHONPATH=. $(DL_ENV)/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 \
			2>&1 | tee logs/api.log; \
	elif [ "$(_COMP)" = "ui" ]; then \
		lsof -ti:5173 2>/dev/null | xargs -r kill -9 2>/dev/null; true; \
		mkdir -p logs; \
		cd ui && npm run dev 2>&1 | tee ../logs/ui.log; \
	elif command -v docker > /dev/null 2>&1 && docker info > /dev/null 2>&1; then \
		if $(DOCKER_RUNNING); then \
			echo "Stack already running.  API → http://localhost:8000  UI → http://localhost:5173"; \
			printf "Restart fresh? [y/N] "; \
			read confirm; \
			if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
				$(COMPOSE) down; \
				lsof -ti:8000 -ti:5173 2>/dev/null | xargs -r kill -9 2>/dev/null; true; \
				if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
					echo "CUDA detected — GPU passthrough enabled"; \
					TORCH_BUILD_TYPE=cu121 $(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up --build -d; \
				else \
					$(COMPOSE) up --build -d; \
				fi; \
				echo "Stack running.  API → http://localhost:8000  UI → http://localhost:5173"; \
				echo "Logs: make logs | make logs api | make logs ui"; \
			else \
				echo "Keeping existing instance."; \
			fi; \
		else \
			lsof -ti:8000 -ti:5173 2>/dev/null | xargs -r kill -9 2>/dev/null; true; \
			if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
				echo "CUDA detected — GPU passthrough enabled"; \
				TORCH_BUILD_TYPE=cu121 $(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up --build -d; \
			else \
				$(COMPOSE) up --build -d; \
			fi; \
			echo "Stack running.  API → http://localhost:8000  UI → http://localhost:5173"; \
			echo "Logs: make logs | make logs api | make logs ui"; \
		fi; \
	else \
		echo ""; \
		echo "  Docker not available — start services in separate terminals:"; \
		echo "    make start api   # API on :8000  (hot-reload, logs → logs/api.log)"; \
		echo "    make start ui    # UI  on :5173  (logs → logs/ui.log)"; \
		echo ""; \
	fi

stop:
	$(COMPOSE) down

# ── Logs (Docker or local, auto-detected) ─────────────────────────────────────
logs:
	@if [ "$(_COMP)" = "api" ]; then \
		if $(DOCKER_RUNNING); then \
			$(COMPOSE) logs -f api; \
		elif [ -f logs/api.log ]; then \
			tail -f logs/api.log; \
		else \
			echo "No API logs found. Start with: make start api"; \
		fi; \
	elif [ "$(_COMP)" = "ui" ]; then \
		if $(DOCKER_RUNNING); then \
			$(COMPOSE) logs -f ui; \
		elif [ -f logs/ui.log ]; then \
			tail -f logs/ui.log; \
		else \
			echo "No UI logs found. Start with: make start ui"; \
		fi; \
	else \
		if $(DOCKER_RUNNING); then \
			$(COMPOSE) logs -f; \
		elif [ -f logs/api.log ] || [ -f logs/ui.log ]; then \
			tail -f logs/api.log logs/ui.log 2>/dev/null; \
		else \
			echo "No running stack found. Start with: make start"; \
		fi; \
	fi

# ── Verify ─────────────────────────────────────────────────────────────────────
verify:
	@if [ "$(_COMP)" = "api" ] || [ "$(_COMP)" = "both" ]; then \
		response=$$(curl -sf http://localhost:8000/health 2>/dev/null); \
		if [ $$? -eq 0 ]; then \
			echo "API  OK   http://localhost:8000/health"; \
			echo "$$response" | python3 -m json.tool; \
		else \
			echo "API  FAIL http://localhost:8000  (not reachable)"; \
		fi; \
	fi
	@if [ "$(_COMP)" = "ui" ] || [ "$(_COMP)" = "both" ]; then \
		code=$$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:5173 2>/dev/null); \
		if [ "$$code" = "200" ]; then \
			echo "UI   OK   http://localhost:5173  (HTTP $$code)"; \
		else \
			echo "UI   FAIL http://localhost:5173  (not reachable)"; \
		fi; \
	fi

# ── Tests & lint ───────────────────────────────────────────────────────────────
test:
	PYTHONPATH=. $(DL_ENV)/python -m pytest tests/ -v

lint:
	$(DL_ENV)/ruff check . && $(DL_ENV)/ruff format --check .

# ── DB ─────────────────────────────────────────────────────────────────────────
migrate:
	PYTHONPATH=. $(DL_ENV)/alembic upgrade head

# ── Spec ───────────────────────────────────────────────────────────────────────
export-spec:
	PYTHONPATH=. $(DL_ENV)/python -c \
		"import json; from api.main import app; open('docs/openapi.json','w').write(json.dumps(app.openapi(), indent=2))"

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean-models:
	@find models/ -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "models/ cleared."
