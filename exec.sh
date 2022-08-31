#!/bin/sh

APPDIR=`dirname $0`
cd ./repo
pip install -r requirements.txt
pip install .
cd docs/tutorials
python $@
return $?

# python -u $APPDIR/main_qm9.py --num_workers 4 $@
# return $?
