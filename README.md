---
author: Tetragramm
title: How to use the LaTeX Catalog Assistant
---

# Introduction

The point of this tool is to make building aircraft catalogs in the
style of the Flying Circus ones easier. You will be able to make fancier
things if you know and understand LaTeX, but it is not necessary to use
this tool.

# Why LaTeX?

LaTeXis basically a programming language for laying out documents,
whether academic papers or books or slides for presentations. It is very
powerful, but very complicated. Additional features can be added by
including packages, basically libraries of layout functions, that can
make complicated things simpler.

For example, mirroring text is not something that can be done in a
normal text editor, but in LaTeX, it is simple. The text ɿoɿɿiM|Mirror is
made using the code

    \reflectbox{Mirror}|Mirror

using the package called \"graphicx\".

Only some packages are available, with the most common commands you may
use described below in the section on aircraft descriptions.

# Setup

First, you must install the dependencies.

-   [TeX Live](https://tug.org/texlive/)

-   The Fonts (provided)

Fortunately, these are nice simple installers, and you should be able to
handle it fine. The TeX Live installer does download a lot of packages,
which can take some time. For slower internet connections, potentially
hours. The fonts are stored in the folder called \"fonts\". On Windows,
just right click on them and hit install.

# What you Get

In this folder are the fonts to install, some template files that are
used during processing, and the Python script Compile.py.
That's it! That's all you need, besides your airplane designs.

# How to Use

1.  Go to the [Plane Builder](https://tetragramm.github.io/PlaneBuilder/index.html)
    and load each of your aircraft designs. Save them using the \"Save
    Catalog\" button at the bottom of the page, in this folder.
2.  Copy Main.tex and rename it the name of your catalog.
3.  Replace the demo calls with the contents of your Saved Catalog files
4.  Fill in the Nicknames, Box Text, and the content for each.
5.  Either run the Compile.py or directly run the command line call inside it.  Be sure to run lualatex 3 times to ensure everything compiles nicely.
1.  Open your PDF and enjoy.

# What to Change

You need to set the name of your catalog and the author information in the InsertFCTitleAndTOC function.

There are two airplanes and one ground vehicle in Main.tex, which need to be replaced with whatever you're actually putting in the catalog.

If you're using Compile.py, you need to change the Title variable to the name of your tex file.

# Basic LaTeX

For all the below, use only what is between the quotes.

To make a paragraph break, include an empty line or "\\par".

To manually place a line break, for example to keep the table from
getting too wide, use "\\\\".

Underline using "\\underline{Text to underline}"

The font used in the plane catalogs does not have a bold face, but
italics are "\\textit{Text to italicize}"

Font size can be changed by using "\\size" where the valid sizes are
tiny, scriptsize, footnotesize, small, normalsize, large, Large, LARGE,
huge, and Huge.

Lists are easy, but there are several types. Check them out
[here.](https://www.overleaf.com/learn/latex/Lists)