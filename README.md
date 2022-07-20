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

-   [Python 3](https://www.python.org/) (Ensure the box
    "Add Python to environment variables" is checked)

-   [TeX Live](https://tug.org/texlive/)

-   The Fonts (provided)

Fortunately, these are nice simple installers, and you should be able to
handle it fine. The TeX Live installer does download a lot of packages,
which can take some time. For slower internet connections, potentially
hours. The fonts are stored in the folder called \"fonts\". On Windows,
just right click on them and hit install.

# What you Get

In this folder are the fonts to install, some template files that are
used during processing, and the Python scripts Create.py and Compile.py.
That's it! That's all you need, besides your airplane designs.

# How to Use

1.  Go to the [Plane Builder](https://tetragramm.github.io/PlaneBuilder/index.html)
    and load each of your aircraft designs. Save them using the \"Save
    Catalog\" button at the bottom of the page, in this folder.

2.  Rename those files so they are in the order you wish them in the
    catalog. I suggest naming them something like \"A_First_Plane.txt\",
    \"B_Second_Plane.txt\", ect.

3.  Open each of them and change the line \"Insert Nickname Here\" to
    the actual nickname of the plane.

4.  Open a command prompt or powershell window in the folder. Hold the
    Shift key and right click in the folder, and click the "Open command
    prompt window here" or "Open PowerShell window here" options.

5.  Alternatively, do this by searching in the start menu, then using
    the command \"cd\" (change directory) to navigate to this folder.

6.  Run the command

                python .\Create.py

    and when asked, type the Title and zero or more authors to include.
    When you are done entering authors, just press enter on an empty
    line.

7.  Wait just a minute as a gigantic pile of text cascades down the
    window. Don't worry, it doesn't matter.

8.  Note the changes to the directory. Three new folders (desc, images,
    subfiles), an AuthorInfo.text, some temporary files created by
    LaTeX, and most importantly, a pdf file!

# What to Change

The file named AuthorInfo.text has the first line as the Title, and each
following line is an author's name. This way you don't need to re-enter
those if you add additional planes and need to re-run Create.py

The folder \"images\" contains the aircraft images, or placeholders. You
can replace them with the images you want. The name of the file is what
is important. For example you may replace \"Basic_Biplane_image.png\"
with \"Basic_Biplane_image.jpg\" with no issues, but trying to use
\"Basic_Biplane.png\" will not work. The images are automatically
resized to fit, but you may wish to make or edit images into the
preferred aspect ratio for better appearances.

The folder \"desc\" contains all of the text you edit. For each plane
there is a \"PlaneName_desc.txt\" and a \"PlaneName_table.txt\".

## The Table

The table is the simpler file. Each row consists of two parts, separated
by an = sign, like the default one, reproduced below.

```
Role=Edit, Add or
Served With=remove lines
First Flight=to fill
Strengths=out the
Weaknesses=table
Inspiration=like this.
```

The part to the left of the equal sign makes up the first column of the
plane's table. The part to the right, the second. Don't put more than
one equal sign per row, or it won't work. You can add or remove rows to
your heart's content, and are not limited to the ones already there,
which were chosen because that's what the first catalog used.

## The Description

The description file is actually a very simple LaTeX file. It has a two
line header, and a one line footer. In-between the \\begin and \\end is
the place where you put your aircraft descriptions. The text within it
will be distributed evenly over the two columns of the page.

Because it is a LaTeX document, you can easily use simple commands, and
with a little effort, more complex formatting. Check out the lovely
tutorial at
[Overleaf](https://overleaf.com/learn/latex/Paragraphs_and_new_lines)
for how to do even complicated formatting. For basic work, see the next
section. When you make changes, you will notice they don't show up in
the PDF. To see the results, you must compile the document. If you have
added, removed, re-ordered an airplane (Or changed values in the save
file from the catalog), you will need to re-run the Create script.

        python .\Create.py

It will read in everything and spit out a fully compiled version without
erasing any of the work you've done. It does replace the contents of
subfiles, and of the main .tex file.

If you have not altered the aircraft, simply run the Compile script.

        python .\Compile.py

Once complete, your file should be ready.

# Basic LaTeX

For all the below, use only what is between the quotes.

To make a paragraph break, include an empty line or "\\par".

To manually place a line break, for example to keep the table from
getting too wide, use "\\\\".

Underline using "\\underline{Text to underline}"

The font used in the plane catalogs does not have a bold face, but
italics are "\\emph{Text to italicize}"

Font size can be changed by using "\\size" where the valid sizes are
tiny, scriptsize, footnotesize, small, normalsize, large, Large, LARGE,
huge, and Huge.

Lists are easy, but there are several types. Check them out
[here.](https://www.overleaf.com/learn/latex/Lists)

# Troubleshooting

Observed problem: Running the python command just prints "Python", and
nothing happens.\
Solution: Run the python installer, choose modify installation, and
ensure the "Add Python to environment variables" box is checked, then
re-open the PowerShell or cmd window.\
If the box is already checked, or if this does not resolve the problem,
type "Get-Command python" If it shows the existence of a python.exe with
version 0.0.0.0, navigate to the location given by Source, and delete
it. This is the Windows Store's attempt to be helpful, but it is
failing.
