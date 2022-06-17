SEED=42 # only used in training, here to identify the training checkpoint path
DOMAIN="ampere"
EXP_NAME=demo-${DOMAIN}_seed=${SEED}

python -m argument_relation_transformer.infer \
  --datadir=./data \
  --dataset=${DOMAIN} \
  --eval-set=test \
  --exp-name=demo-${DOMAIN}_seed=${SEED} \
  --ckptdir=./checkpoints/ \
  --batch-size=32
