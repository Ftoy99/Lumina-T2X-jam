#!/bin/bash


# Lumina-Next supports any resolution (up to 2K)
# res="1024:024x1024 1536:1536x1536 1664:1664x1664 1792:1792x1792 2048:2048x2048"
res=512:512x512
t=4
cfg=4.0
seed=69
steps=20
solver=euler
model_dir=Lumina-Next-SFT
cap_dir=captions.txt
out_dir=output
python3 -u sample_mf.py --ckpt ${model_dir} \
--image_save_path ${out_dir} \
--solver ${solver} --num_sampling_steps ${steps} \
--caption_path ${cap_dir} \
--seed ${seed} \
--resolution ${res} \
--time_shifting_factor ${t} \
--cfg_scale ${cfg} \
--batch_size 1 \
--use_flash_attn False # You can set this to False if you want to disable the flash attention
