python train.py --gpu_ids 0 --num_classes 9 --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --sam_ckpt /home/yanzhonghao/data/models/segment/sam/sam_vit_b_01ec64.pth --model_type lora --batch_size 12 --num_workers 12 --loss ce --lr 3e-3 --dice_weight 0.8

python train.py --gpu_ids 0 --num_classes 9 --dataset sabs_sammed --scale 0.05 --sam_mode vit_b --sam_ckpt /home/yanzhonghao/data/models/segment/sam/sam_vit_b_01ec64.pth --model_type lora --batch_size 12 --num_workers 12 --loss ce --lr 3e-3 --dice_weight 0.8

python train.py --gpu_ids 0 --num_classes 9 --dataset sabs_sammed --scale 0.01 --sam_mode vit_b --sam_ckpt /home/yanzhonghao/data/models/segment/sam/sam_vit_b_01ec64.pth --model_type lora --batch_size 12 --num_workers 12 --loss ce --lr 3e-3 --dice_weight 0.8
