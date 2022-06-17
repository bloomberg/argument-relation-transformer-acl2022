## Argument Structure Prediction

Code release for paper `Efficient Argument Structure Extraction with Transfer Learning and Active Learning`

```bibtex
@inproceedings{hua-wang-2022-efficient,
    title = "Efficient Argument Structure Extraction with Transfer Learning and Active Learning",
    author = "Hua, Xinyu  and
      Wang, Lu",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.36",
    pages = "423--437",
}
```

## Requirements

The original project is tested under the following environments:

```
python==3.7.12
torch==1.6.0
pytorch_lightning==1.5.0
transformers==4.10.3
numpy==1.21.6
scikit-learn==1.0.2
```

## Data

We release the AMPERE++ dataset in this [link](https://zenodo.org/record/6362430#.YjJJUprMIba). 
Please download the jsonl files and store under `./data`.


The other four datasets can be downloaded using the links below (requires format conversion, code can be found in `./scripts/`):

- Essays (Stab and Gurevych, 2017): [link](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422)
- AbstRCT (Mayer et al., 2020): [link](https://gitlab.com/tomaye/abstrct/)
- ECHR (Poudyal et al., 2020): [link](http://www.di.uevora.pt/~pq/echr/)
- CDCP (Park and Cardie, 2018; Niculae et al., 2017): [link](https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip)

## Quick Start

First, install the package:

```shell script
pip install -e .
```

To train a standard supervised relation extraction model on AMPERE++:

```shell script
SEED=42
DOMAIN="ampere"

python -m argument_relation_transformer.train \
  --datadir=./data \
  --seed=${SEED} \
  --dataset=${DOMAIN} \
  --ckptdir=./checkpoints \
  --exp-name=demo-${DOMAIN}_seed=${SEED} \
  --warmup-steps=5000 \
  --learning-rate=1e-5 \
  --scheduler=constant \
  --max-epochs=15
```

The trained model will be saved at `./checkpoints/demo-ampere_seed=42/`, the tensorboard metrics can be found under `./tb_logs/demo-ampere_seed=42/` which can be loaded for evaluation.

```shell script
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
```

The prediction results will be saved to `./outputs/demo-ampere_seed=42.jsonl`, the evaluation metrics will be saved to `./outputs/demo-ampere_seed=42.jsonl.scores`

## Active Learning (AL)

We simulate the pool-based AL, where the entire process consists of 10 iterations. During each iteration, 500 samples are collected based 
on certain sampling strategy. We use the following script to demonstrate this procedure (`train_active_demo.sh`):

```shell script
INTERVAL=500
SEED=42
METHOD="max_entropy"
DOMAIN="ampere"
EXP_NAME="${METHOD}-demo_seed=${SEED}"
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
    n_p=$(( $p + $INTERVAL))
    train ${n_p}
done
```
