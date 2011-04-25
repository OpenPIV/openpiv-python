#!/bin/bash
pyuic4 data/ui/mainwindow.ui -o src/ui/ui_mainwindow.py
pyrcc4 data/ui_resources.qrc -o src/ui/ui_resources_rc.py
