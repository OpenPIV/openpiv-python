#!/usr/bin/python
""" Python script to build Sphinx documents in _build/html/
copy it to ../gh-pages subfolder of the openpiv-python project folder
commit and push it using git to the Github pages
"""


__licence_ = """
Copyright (C) 2011  www.openpiv.net

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

