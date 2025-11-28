accelerate launch diffusion_trainer.py \
        --data_dir "/mnt/f/datasets/nuscenes_lol" \
        --output_dir "runs" \
        --batch_size 4 \
        --resolution 256 \
        --num_workers 8 \