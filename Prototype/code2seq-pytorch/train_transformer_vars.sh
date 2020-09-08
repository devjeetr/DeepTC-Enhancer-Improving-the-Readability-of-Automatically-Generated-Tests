
python main.py  \
--use-bpe \
--train-path data/jtec-var-name/train.c2s \
--val-path data/jtec-var-name/val.c2s \
--test-path data/jtec-var-name/val.c2s \
--train-line-cache data/jtec-var-name/train-line-cache.pkl \
--val-line-cache data/jtec-var-name/val-line-cache.pkl \
--test-line-cache data/jtec-var-name/val-line-cache.pkl \
--subtoken-vocab data/jtec-var-name/subtoken.bpe-vocab.json \
--subtoken-merges data/jtec-var-name/subtoken.bpe-merges.txt \
--target-vocab data/jtec-var-name/target.bpe-vocab.json \
--target-merges data/jtec-var-name/target.bpe-merges.txt \
--node-dict data/jtec-var-name/node_dict.pkl \
--gpus 1 \
--predict-variables \
--shuffle-batches \
--d_model 512 \
--context-encoder-dropout 0.5 \
--transformer-feedforward 512 \
--gradient_clip_val 5.0 \
--dataloader-num-workers 12 \
--transformer-dropout 0.2 \
--batch-size 512 \
--accumulate_grad_batches 2 \
--noam-warmup-steps 4000 \
--check_val_every_n_epoch=1 \
--precision 16 \
# --fast_dev_run true --progress_bar_refresh_rate 0
