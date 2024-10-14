CUDA_VISIBLE_DEVICES=2 python3  tdeepsc_main.py \
    --model  TDeepSC_vqa_model  \
    --output_dir ckpt_record  \
    --batch_size 30 \
    --input_size 224 \
    --lr  1e-4 \
    --epochs 200  \
    --opt_betas 0.95 0.99  \
    --save_freq 2   \
    --ta_perform vqa  
  


CUDA_VISIBLE_DEVICES=1 python3  tdeepsc_main.py --model  TDeepSC_textc_model  --output_dir ckpt_record --batch_size 40 --input_size 32 --lr  3e-5 --epochs 40  --opt_betas 0.95 0.99  --save_freq 2  --ta_perform textc

CUDA_VISIBLE_DEVICES=1 python3  tdeepsc_main.py --model  TDeepSC_vqa_model  --output_dir ckpt_record  --batch_size 30 --input_size 224 --lr  1e-4 --epochs 200  --opt_betas 0.95 0.99  --save_freq 2    --ta_perform vqa