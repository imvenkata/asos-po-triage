.PHONY: install test eval eval-offline audit doctor api demo clean

install:
	python3 -m venv .venv
	./.venv/bin/pip install -e ".[dev]"

test:
	./.venv/bin/python -m pytest -q

eval:
	./.venv/bin/python evals/run_evals.py --json eval_report.json


eval-offline:
	TRIAGE_LLM_PROVIDER=scripted ./.venv/bin/python evals/run_evals.py

audit:
	./.venv/bin/triage audit

doctor:
	./.venv/bin/triage doctor

api:
	./.venv/bin/uvicorn triage.api:app --reload --port 8000

demo:
	./.venv/bin/triage ask "PO-10342 has come in about 12% under the original order value. Can I amend it in place?" --trace

clean:
	rm -rf .venv .pytest_cache eval_report.json
	find . -name __pycache__ -type d -exec rm -rf {} +
