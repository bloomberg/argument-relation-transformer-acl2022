INTERVAL=500
SEED=42
METHOD="max-entropy"
DOMAIN="ampere"
EXP_NAME="${METHOD}-demo_SEED=${SEED}"
CKPTDIR="./checkpoints_al/"
MAX_EPOCHS=10

active() {
# load model from `model-path` (if needed)
# select INTERVAL unlabeled samples
# save to `/tmp/checkpoints_al/[data]/[exp_name]/[method]_[interval].jsonl`, which is
# the ids of **all** labeled data
# args: (1) current sample size;
    python -m argument_relation_transformer.active \
        --dataset=${DOMAIN} \
        --datadir="./data/" \
        --ckptdir=${CKPTDIR} \
        --exp-name=${EXP_NAME} \
        --method=${METHOD} \
        --seed=${SEED} \
        --interval=${INTERVAL} \
        --huggingface-path="./huggingface/" \
        --current-sample-size=$1
}

train() {
# load data from `{ckptdir}/{exp-name}/{method}_{current-sample-size}.jsonl`
# train the model and save to `{ckptdir}/{exp-name}/model_{current-sample-size}/`
    python -m argument_relation_transformer.train \
        --datadir=./data \
        --seed=${SEED} \
        --dataset=${DOMAIN} \
        --ckptdir=${CKPTDIR} \
        --exp-name=${EXP_NAME} \
        --warmup-steps=500 \
        --learning-rate=1e-5 \
        --huggingface-path="./huggingface/" \
        --scheduler-type=constant \
        --max-epochs=${MAX_EPOCHS} \
        --from-al \
        --al-method=${METHOD} \
        --current-sample-size=$1
}

for p in 0 500 1000 1500 2000 2500 3000 3500 4000 4500;
do
    # each step, we have p samples already, and are selecting the next 500 samples
    active ${p}
    n_p=$(( $p + $INTERVAL ))
    train ${n_p}
done
