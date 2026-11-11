.PHONY: help run stop run-api run-ui logs logs-api logs-ui verify verify-api verify-ui test lint migrate export-spec clean-models env env-gpu dataset

COMPOSE = docker compose
DL_ENV  = /home/hhaniffa/miniconda/envs/afd_v2/bin

# True when the API Docker container is currently running
DOCKER_API_RUNNING = docker compose ps --services --filter status=running 2>/dev/null | grep -q api

# ── Help (default target) ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Anomaly Flow Detector — available commands"
	@echo ""
	@echo "  Setup (run once)"
	@echo "    make env          Create afd_v2 env + deps; auto-installs GPU torch if CUDA found"
	@echo "    make env-gpu      Manually upgrade torch to CUDA 12.1 (RTX A2000)"
	@echo "    make dataset      Download CIC-IDS2017 (~225 MB); skips if already present"
	@echo ""
	@echo "  Stack"
	@echo "    make run          Start the stack — Docker if available, local otherwise"
	@echo "    make stop         Stop the Docker stack"
	@echo "    make run-api      Run API locally on :8000 (hot-reload, logs → logs/api.log)"
	@echo "    make run-ui       Start Vite dev server on :5173  (logs → logs/ui.log)"
	@echo ""
	@echo "  Logs (auto-detects Docker or local)"
	@echo "    make logs         Follow all logs"
	@echo "    make logs-api     Follow API logs"
	@echo "    make logs-ui      Follow UI logs"
	@echo ""
	@echo "  Verify"
	@echo "    make verify       Check API and UI are reachable"
	@echo "    make verify-api   Check API /health endpoint"
	@echo "    make verify-ui    Check UI is serving on :5173"
	@echo ""
	@echo "  Development"
	@echo "    make test         Run pytest suite"
	@echo "    make lint         Ruff check + format check"
	@echo "    make migrate      Apply Alembic migrations"
	@echo "    make export-spec  Write docs/openapi.json"
	@echo "    make clean-models Delete versioned model dirs under models/"
	@echo ""

# ── Stack ──────────────────────────────────────────────────────────────────────
run:
	@if command -v docker > /dev/null 2>&1 && docker info > /dev/null 2>&1; then \
		if $(DOCKER_API_RUNNING); then \
			echo "Stack is already running.  API → http://localhost:8000  UI → http://localhost:5173"; \
			printf "Restart fresh? This will stop the current instance. [y/N] "; \
			read confirm; \
			if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
				$(COMPOSE) down; \
				lsof -ti:8000 -ti:5173 2>/dev/null | xargs -r kill -9 2>/dev/null; true; \
				if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
					echo "CUDA detected — building with cu121 torch"; \
					TORCH_BUILD_TYPE=cu121 $(COMPOSE) up --build -d; \
				else \
					echo "No CUDA — building with CPU torch"; \
					$(COMPOSE) up --build -d; \
				fi; \
				echo "Stack running.  API → http://localhost:8000  UI → http://localhost:5173"; \
				echo "Logs: make logs | make logs-api | make logs-ui"; \
			else \
				echo "Keeping existing instance."; \
			fi; \
		else \
			lsof -ti:8000 -ti:5173 2>/dev/null | xargs -r kill -9 2>/dev/null; true; \
			if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
				echo "CUDA detected — building with cu121 torch"; \
				TORCH_BUILD_TYPE=cu121 $(COMPOSE) up --build -d; \
			else \
				echo "No CUDA — building with CPU torch"; \
				$(COMPOSE) up --build -d; \
			fi; \
			echo "Stack running.  API → http://localhost:8000  UI → http://localhost:5173"; \
			echo "Logs: make logs | make logs-api | make logs-ui"; \
		fi; \
	else \
		echo ""; \
		echo "  Docker not available — local runtime mode."; \
		echo ""; \
		echo "  If afd_v2 env is not yet set up, run first:"; \
		echo "    make env"; \
		echo ""; \
		echo "  Then start each service in a separate terminal:"; \
		echo "    make run-api    # API on :8000  (logs → logs/api.log)"; \
		echo "    make run-ui     # UI  on :5173  (logs → logs/ui.log)"; \
		echo ""; \
		echo "  Follow logs with: make logs-api | make logs-ui"; \
		echo ""; \
	fi

stop:
	$(COMPOSE) down

run-api:
	@lsof -ti:8000 2>/dev/null | xargs -r kill -9 2>/dev/null; true
	@mkdir -p logs
	PYTHONPATH=. $(DL_ENV)/alembic upgrade head
	PYTHONPATH=. $(DL_ENV)/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 \
		2>&1 | tee logs/api.log

run-ui:
	@lsof -ti:5173 2>/dev/null | xargs -r kill -9 2>/dev/null; true
	@mkdir -p logs
	cd ui && npm run dev 2>&1 | tee ../logs/ui.log

# ── Logs (Docker or local, auto-detected) ─────────────────────────────────────
logs:
	@if $(DOCKER_API_RUNNING); then \
		$(COMPOSE) logs -f; \
	elif [ -f logs/api.log ] || [ -f logs/ui.log ]; then \
		tail -f logs/api.log logs/ui.log 2>/dev/null; \
	else \
		echo "No running stack found. Start with: make run"; \
	fi

logs-api:
	@if $(DOCKER_API_RUNNING); then \
		$(COMPOSE) logs -f api; \
	elif [ -f logs/api.log ]; then \
		tail -f logs/api.log; \
	else \
		echo "No API logs found. Start the API with: make run-api"; \
	fi

logs-ui:
	@if $(DOCKER_API_RUNNING); then \
		$(COMPOSE) logs -f ui; \
	elif [ -f logs/ui.log ]; then \
		tail -f logs/ui.log; \
	else \
		echo "No UI logs found. Start the UI with: make run-ui"; \
	fi

# ── Dev environment ────────────────────────────────────────────────────────────
env:
	@conda env list | grep -q "^afd_v2 " \
		&& echo "afd_v2 already exists — skipping create" \
		|| conda create -n afd_v2 python=3.12 -y
	conda run -n afd_v2 pip install -r requirements.txt
	@if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then \
		echo "CUDA detected — installing cu121 torch"; \
		conda run -n afd_v2 pip install torch==2.5.1 \
			--index-url https://download.pytorch.org/whl/cu121; \
	else \
		echo "No CUDA detected — using CPU torch (run make env-gpu to upgrade later)"; \
	fi

env-gpu:
	conda run -n afd_v2 pip install torch==2.5.1 \
		--index-url https://download.pytorch.org/whl/cu121

dataset:
	@CSV_COUNT=$$(find -L data/CIC-IDS2017 -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l); \
	if [ "$$CSV_COUNT" -gt 0 ]; then \
		echo "CIC-IDS2017 already present ($$CSV_COUNT CSV files) — skipping download."; \
	else \
		mkdir -p data/CIC-IDS2017; \
		echo "Downloading CIC-IDS2017 (~225 MB) …"; \
		conda run -n afd_v2 gdown --id 10UG7eEAUI8kV4_CMFsrK-5nVz4Bm0Ey3 -O data/CIC-IDS2017.zip; \
		unzip -q data/CIC-IDS2017.zip -d data/CIC-IDS2017/; \
		rm data/CIC-IDS2017.zip; \
		echo "Done — $$(find -L data/CIC-IDS2017 -maxdepth 1 -name '*.csv' | wc -l) CSV files ready."; \
	fi

# ── Verify ─────────────────────────────────────────────────────────────────────
verify: verify-api verify-ui

verify-api:
	@response=$$(curl -sf http://localhost:8000/health 2>/dev/null); \
	if [ $$? -eq 0 ]; then \
		echo "API  OK  http://localhost:8000/health"; \
		echo "$$response" | python3 -m json.tool; \
	else \
		echo "API  UNREACHABLE  http://localhost:8000"; \
	fi

verify-ui:
	@code=$$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:5173 2>/dev/null); \
	if [ "$$code" = "200" ]; then \
		echo "UI   OK  http://localhost:5173  (HTTP $$code)"; \
	else \
		echo "UI   UNREACHABLE  http://localhost:5173"; \
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
