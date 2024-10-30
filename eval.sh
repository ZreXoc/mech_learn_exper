# /bin/bash
workers=4

mode=eval
lr=3e-5
batch_size=40
epochs=20
save_freq=1
#pretrained=hfl/chinese-bert-wwm
pretrained=cache/FinBERT/FinBERT_L-12_H-768_A-12_pytorch

id=$(date "+%m%d%H%M")

export TRANSFORMERS_OFFLINE=1

python ./eval.py --id=$id --mode=$mode --pretrained=$pretrained --lr=$lr --batch_size=$batch_size --niter=$epochs --workers=$workers --save_freq=$save_freq --debug --log
