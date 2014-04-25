#!/bin/bash
pyside-uic openpiv/data/ui/mainwindow.ui -o openpiv/ui/ui_mainwindow.py
pyrcc4 openpiv/data/ui_resources.qrc -o openpiv/ui/ui_resources_rc.py
