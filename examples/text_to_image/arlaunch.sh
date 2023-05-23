export MODEL_NAME="CompVis/stable-diffusion-v1-4"
#export dataset_name="lambdalabs/pokemon-blip-captions"

BATCH_SIZE=16
LEARNING_RATE=1e-5

# NOTE: the one we trained already had height and width flipped
# So should be height of 512 and width of 768


#WIDTH=512
#HEIGHT=512

# TODO try input perturbation?

accelerate launch --main_process_port 29051 --mixed_precision="fp16"  --multi_gpu artrain.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_csv='/persist/ljohnston/datasets/may17_finetuning/ar_train.csv' \
  --dataloader_num_workers=32 \
  --use_ema \
  --train_batch_size=${BATCH_SIZE} \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=1000000 \
  --learning_rate=${LEARNING_RATE} \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=1000 \
  --output_dir="may22_ar_lr${LEARNING_RATE}_b${BATCH_SIZE}_2"  \

  # TODO scale_lr?
  #--scale_lr  
  #--random_flip

  # TODO why does adding to dataloader_num_workers make the training slower??
