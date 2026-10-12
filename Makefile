PYTHON ?= .venv/bin/python

.PHONY: all install test figures notebooks report slides clean

all: test figures report slides

install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

figures:
	$(PYTHON) scripts/make_figures.py

notebooks:
	$(PYTHON) scripts/build_notebooks.py

report:
	cd report && latexmk -pdf -interaction=nonstopmode report.tex

slides:
	cd Presentation && latexmk -pdf -interaction=nonstopmode slides.tex

clean:
	cd report && latexmk -c report.tex
	cd Presentation && latexmk -c slides.tex
