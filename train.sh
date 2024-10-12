# /bin/bash
workers=4

mode=train
lr=3e-5
batch_size=40
epochs=20
save_freq=1
pretrained=hfl/chinese-bert-wwm

id=$(date "+%m%d%H%M")

export TRANSFORMERS_OFFLINE=1

python ./train2.py --id=$id --mode=$mode --pretrained=$pretrained --lr=$lr --batch_size=$batch_size --save_freq=$save_freq --workers=$workers --save_freq=$save_freq --debug --log
