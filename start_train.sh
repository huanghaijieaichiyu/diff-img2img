accelerate launch diffusion_trainer.py \
        --data_dir "your data" \
        --epochs 50 \
        --batch_size 4 \
        --resolution 256 \
        --num_workers 8 \
        --output_dir "runs/retinex" \
        --use_retinex \
        --mixed_precision="fp16" \
        --prediction_type="v_prediction" \