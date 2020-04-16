#!/bin/bash         

gepetto-gui &
hpp-rbprm-server &
ipython -i --no-confirm-exit ./lp_urdfs_path.py ./$1

pkill -f  'gepetto-gui'
pkill -f  'hpp-rbprm-server'
