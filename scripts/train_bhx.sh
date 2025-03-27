python train.py --gpu_ids 0 --num_classes 5 --dataset bhx_sammed --scale 0.1 --model_type lora --batch_size 16 --num_workers 16 --loss ce --lr 1e-3 --dice_weight 0.8

python train.py --gpu_ids 0 --num_classes 5 --dataset bhx_sammed --scale 0.05 --model_type lora --batch_size 16 --num_workers 16 --loss ce --lr 1e-3 --dice_weight 0.8

python train.py --gpu_ids 0 --num_classes 5 --dataset bhx_sammed --scale 0.01 --model_type lora --batch_size 16 --num_workers 16 --loss ce --lr 1e-3 --dice_weight 0.8

