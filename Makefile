.PHONY: setup test test-fast lint format demo serve web-install web-build web-dev clean-data docker-build

UV ?= uv

setup:
	$(UV) sync --all-extras
	@echo "Python environment ready. Run 'make web-install' for the frontend."

lint:
	$(UV) run ruff check src tests scripts
	$(UV) run ruff format --check src tests scripts

format:
	$(UV) run ruff format src tests scripts
	$(UV) run ruff check --fix src tests scripts

test:
	$(UV) run pytest -q

test-fast:
	$(UV) run pytest -q -m "not slow and not network"

test-offline:
	AERODOPPLER_OFFLINE=true $(UV) run pytest -q -m "not network"

demo:
	AERODOPPLER_OFFLINE=true $(UV) run aerodoppler demo --out data/runs/demo

serve:
	$(UV) run aerodoppler serve --host 127.0.0.1 --port 8000

web-install:
	cd web && npm install

web-build:
	cd web && npm run build

web-dev:
	cd web && npm run dev

docker-build:
	docker build -t aerodoppler:local .

clean-data:
	@echo "This deletes ALL local recordings, derived data and runs under ./data"; \
	read -p "Type 'purge' to confirm: " c; [ "$$c" = "purge" ] && rm -rf data/raw data/normalized data/recordings data/runs data/aerodoppler.sqlite3 || echo aborted
