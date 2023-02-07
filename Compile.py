import sys
import re
import os

Title = 'Main'
if os.path.isfile(Title+'.tex'):
    os.system(
        f'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -pdf "{Title}.tex"')
    os.system(
        f'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -pdf "{Title}.tex"')
    os.system(
        f'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -pdf "{Title}.tex"')
else:
    print('No LaTeX file matching Title found.')
