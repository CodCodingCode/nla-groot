# Workshop short paper (MechInterp / NeurIPS 2026 target).
DOC := main
LATEXMK := $(shell command -v latexmk 2>/dev/null)

.PHONY: all clean watch view

all:
ifeq ($(LATEXMK),)
	pdflatex -interaction=nonstopmode $(DOC).tex || true
	bibtex $(DOC) || true
	pdflatex -interaction=nonstopmode $(DOC).tex || true
	pdflatex -interaction=nonstopmode $(DOC).tex
else
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(DOC).tex
endif

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(DOC).tex

clean:
	rm -f $(DOC).aux $(DOC).bbl $(DOC).blg $(DOC).log $(DOC).out \
	      $(DOC).toc $(DOC).fls $(DOC).fdb_latexmk $(DOC).synctex.gz \
	      $(DOC).run.xml $(DOC).bcf

distclean: clean
	rm -f $(DOC).pdf
