# DOMAIN expects one of ['ampere', 'ukp', 'cdcp', 'abst_rct', 'echr'], data need to be downloaded separately
DOMAIN="ampere"
SEED=42
python -m argument_relation_transformer.train \
  --datadir=./data \
  --seed=${SEED} \
  --dataset=${DOMAIN} \
  --ckptdir=./checkpoints \
  --exp-name=demo-${DOMAIN}_seed=${SEED} \
  --warmup-steps=5000 \
  --learning-rate=1e-5 \
  --huggingface-path=./huggingface/ \
  --scheduler-type=constant \
  --max-epochs=15
