export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task anomaly_detection \
  --dataset geolife \
  --abnormal_samples 1 \
  --normal_samples 9 \
  --abnormal_index 1 \
  --random_seed 42 \
  --batch_size 10 \
  --epochs 10  \
  --lr 0.0001 \
  --num_classes 2 \
  --num_layers 3 \
  --dim_hidden 128 \
  --num_kernels 8 \
  --input_dim_st 4 \
  --input_dim_text 768 \
  --alpha 0.001 \
  --scaler 0.25 \
  --num_heads 1 \
  --dropout 0.5 \
  --attn_threhold 0.1 \
  --return_attention false \
