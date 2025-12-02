accelerate launch diffusion_trainer.py \
        --data_dir "/mnt/f/datasets/kitti_LOL" \
        --epochs 30 \
        --batch_size 4 \
        --resolution 256 \
        --num_workers 8 \
        --output_dir "runs/retinex" \
        --use_retinex \
        --mixed_precision="fp16" \
        --prediction_type="v_prediction" \