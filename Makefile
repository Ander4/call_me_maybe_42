ARGS=
DEBUGGER= -m pdb

MYPYFLAGS=--warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

all: run

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python $(DEBUGGER) -m src

clean:
	find src -depth -type d -name "__pycache__" -exec rm -rf {} \;
	find src -depth -type d -name ".pytest_cache" -exec rm -rf {} \;
	find src -depth -type d -name ".mypy_cache" -exec rm -rf {} \;

lint:
	flake8 src
	mypy src  $(MYPYFLAGS)

lint-strict:
	flake8 src
	mypy src --strict

.SILENT: all run debug clean lint lint-strict