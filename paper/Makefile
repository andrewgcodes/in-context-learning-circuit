LATEX = pdflatex -interaction=nonstopmode -halt-on-error

main.pdf: main.tex refs.bib figures/*.pdf
	$(LATEX) main.tex
	bibtex main
	$(LATEX) main.tex
	$(LATEX) main.tex

arxiv: main.pdf
	rm -rf arxiv arxiv.tar.gz && mkdir -p arxiv/figures
	cp main.tex main.bbl refs.bib iclr2024_conference_mod.sty iclr2024_conference.bst fancyhdr.sty arxiv/
	cp figures/*.pdf arxiv/figures/
	tar czf arxiv.tar.gz -C arxiv .
	@echo "arxiv.tar.gz ready (pdflatex + bibtex; main.bbl included)"

clean:
	rm -rf arxiv arxiv.tar.gz main.aux main.log main.out main.pdf main.bbl main.blg main.toc

.PHONY: arxiv clean
