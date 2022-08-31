#!/bin/sh

APPDIR=`dirname $0`
pip install -r requirements.txt
pip install .
cd docs/tutorials
python train_painn_lumo.py

# python -u $APPDIR/main_qm9.py --num_workers 4 $@
# return $?