python train.py --gpu_ids 0 --num_classes 9 --dataset sabs_sammed --scale 0.1 --model_type lora --batch_size 12 --num_workers 12 --loss ce --lr 3e-3 --dice_weight 0.8

python train.py --gpu_ids 0 --num_classes 9 --dataset sabs_sammed --scale 0.05 --model_type lora --batch_size 12 --num_workers 12 --loss ce --lr 3e-3 --dice_weight 0.8

python train.py --gpu_ids 0 --num_classes 9 --dataset sabs_sammed --scale 0.01 --model_type lora --batch_size 12 --num_workers 12 --loss ce --lr 3e-3 --dice_weight 0.8
