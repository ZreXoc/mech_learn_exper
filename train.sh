# /bin/bash
python ./train.py | tee $(date "+%d%H%M").log
