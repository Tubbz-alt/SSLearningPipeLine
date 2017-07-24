# Shell script to handle setup.

source /reg/g/psdm/etc/psconda.sh

PYTHONPATH=../transferLearning/pylabelme python user_driver.py $@

