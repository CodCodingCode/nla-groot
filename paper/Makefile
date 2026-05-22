# Workshop short paper (MechInterp / NeurIPS 2026 target).
# CoRL 2026: make corl && make docx
DOC := main
CORL_DOC := main_corl
LATEXMK := $(shell command -v latexmk 2>/dev/null)

.PHONY: all corl abstract docx figure clean watch view

all:
ifeq ($(LATEXMK),)
	pdflatex -interaction=nonstopmode $(DOC).tex || true
	bibtex $(DOC) || true
	pdflatex -interaction=nonstopmode $(DOC).tex || true
	pdflatex -interaction=nonstopmode $(DOC).tex
else
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(DOC).tex
endif

abstract:
	cd figures && pdflatex -interaction=nonstopmode render_figure.tex
	cd figures && pdftoppm -png -singlefile -r 200 render_figure.pdf system_overview 2>/dev/null || true
	pdflatex -interaction=nonstopmode abstract_corl.tex
	pdflatex -interaction=nonstopmode abstract_corl.tex
	python3 build_abstract_docx.py

corl:
ifeq ($(LATEXMK),)
	pdflatex -interaction=nonstopmode $(CORL_DOC).tex || true
	bibtex $(CORL_DOC) || true
	pdflatex -interaction=nonstopmode $(CORL_DOC).tex || true
	pdflatex -interaction=nonstopmode $(CORL_DOC).tex
else
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(CORL_DOC).tex
endif

figure:
	cd figures && pdflatex -interaction=nonstopmode render_figure.tex
	cd figures && pdftoppm -png -singlefile -r 200 render_figure.pdf system_overview || \
	  convert -density 200 render_figure.pdf figures/system_overview.png

docx: figure
	python3 build_docx.py

watch:
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode $(DOC).tex

clean:
	rm -f $(DOC).aux $(DOC).bbl $(DOC).blg $(DOC).log $(DOC).out \
	      $(DOC).toc $(DOC).fls $(DOC).fdb_latexmk $(DOC).synctex.gz \
	      $(DOC).run.xml $(DOC).bcf

distclean: clean
	rm -f $(DOC).pdf $(CORL_DOC).pdf abstract_corl.pdf nla_groot_corl2026.docx \
	      nla_groot_abstract_corl2026.docx figures/system_overview.png \
	      figures/render_figure.pdf figures/render_figure.aux figures/render_figure.log \
	      abstract_corl.aux abstract_corl.log
