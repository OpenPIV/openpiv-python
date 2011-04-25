#!/usr/bin/python
""" Python script to build Sphinx documents in _build/html/
copy it to ../gh-pages subfolder of the openpiv-python project folder
commit and push it using git to the Github pages
"""
import os
from subprocess import call

# build the sphinx doc
call(["make","html"])

# copy the files and directories
# first find the directory gh-pages
curpath = os.getcwd()
gh_pages_path = os.path.join(curpath[:curpath.find('master')],'gh-pages')

if os.path.isdir(gh_pages_path):
    call(['cp','-R', '_build/html/', gh_pages_path])
    os.chdir(gh_pages_path)
    call(['git','add','.'])
    call(['git','commit','-m','"updated documents"'])
    call(['git','push','origin','gh-pages'])

