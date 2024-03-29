#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=gen-KAIROS-info-simtr

rm -rf preprocessed_*
rm -rf checkpoints/${CKPT_NAME}
# does not use informative mentions 
python dlee_train_test_init.py --model=constrained-gen --ckpt_name=${CKPT_NAME} \
    --dataset=KAIROS \
    --train_file=data/wikievents/train_info.jsonl \
    --val_file=data/wikievents/dev_info.jsonl \
    --test_file=data/wikievents/test_info.jsonl \
    --train_batch_size=2 \
    --eval_batch_size=1 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4\
    --num_train_epochs=5\
    --mark_trigger \
    --coref_dir=data/wikievents/coref \
    --use_info \
    --sim_train
