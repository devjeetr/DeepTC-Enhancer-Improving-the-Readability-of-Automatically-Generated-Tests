echo $(which python)

python main.py  \
--use-bpe \
--dict-path ../data/dict.c2s \
--train-path ../data/train.c2s \
--val-path ../data/val.c2s \
--test-path ../data/test.c2s \
--train-line-cache ../data/train-line-cache.pkl \
--val-line-cache ../data/val-line-cache.pkl \
--test-line-cache ../data/test-line-cache.pkl \
--subtoken-vocab ../data/subtoken.bpe-vocab.json \
--subtoken-merges ../data/subtoken.bpe-merges.txt \
--target-vocab ../data/target.bpe-vocab.json \
--target-merges ../data/target.bpe-merges.txt \
--node-dict ../data/node_dict.pkl \
--target-len 7 \
--subtoken-len 9 \
--gpus 1 \
--shuffle-batches \
--d_model 320 \
--transformer-feedforward 320 \
--transformer-dropout 0.5 \
--gradient_clip_val 5.0 \
--dataloader-num-workers 8 \
--batch-size 256 \
--accumulate_grad_batches=4
# --fast_dev_run true --progress_bar_refresh_rate 0
