all: standalone clean

standalone:
	python results.py
	-find . -maxdepth 1 -name '*.tex' ! -name 'tikz_settings.tex' -print0 | xargs -0 -I {} latexmk -f -pdf -quiet -interaction=nonstopmode {}

clean:
	latexmk -c

clean-all:
	latexmk -c
	find . -maxdepth 1 -name '*.tex' ! -name 'tikz_settings.tex' -exec rm {} \;