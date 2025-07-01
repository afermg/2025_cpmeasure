
# Table of Contents

1.  [Repo contents:](#org1146dbb)
    1.  [For reproducibility:](#org5da93c6)



<a id="org1146dbb"></a>

# Repo contents:

-   draft.org: main draft of the paper
-   bib file with citations
-   Some sty files required by CodeML
-   gen\_pdf.org: Org mode file that allows for deterimnistic generation of the pdf using the makefile
-   Makefile: Automates generation and update of the PDF.


<a id="org5da93c6"></a>

## For reproducibility:

-   flake.{nix,lock}: Contains emacs and uv dependencies for Python figure generation and emacs-based org->tex->PDF
-   `src`: code content. If applicable dependencies will be inside folders within a pyproject.toml or in the header of a script to be run with `uv run --script`

