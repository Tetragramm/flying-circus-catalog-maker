import sys
import re
import os
if os.path.isfile('./AuthorInfo.text'):
    with open('./AuthorInfo.text', 'r') as AI:
        Title = AI.readline()
        Title = Title.strip()
        if os.path.isfile(Title+'.tex'):
            os.system(
                f'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -pdf "{Title}.tex"')
            os.system(
                f'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -pdf "{Title}.tex"')
            os.system(
                f'lualatex -synctex=1 -interaction=nonstopmode -file-line-error -pdf "{Title}.tex"')
        else:
            print('No LaTeX file matching saved Title found.')
else:
    print('No saved AuthorInfo.text found.  Cannot compile.')
