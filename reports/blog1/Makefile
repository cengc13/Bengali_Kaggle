main:
	pdflatex --shell-escape blog1
	bibtex blog1
	pdflatex --shell-escape blog1
	pdflatex --shell-escape blog1

clean:
	rm -rf blog1.aux blog1.log blog1.out blog1.toc _minted-blog1

fast:
	pdflatex --shell-escape blog1
