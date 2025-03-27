# PGP-SAM

## Abstract
The Segment Anything Model (SAM) has demonstrated strong and versatile segmentation capabilities, along with intuitive prompt-based interactions. However, customizing SAM for medical image segmentation requires massive amounts of pixel-level annotations and precise point- or box-based prompt designs. To address these challenges, we introduce PGP-SAM, a novel prototype-based few-shot tuning approach that uses limited samples to replace tedious manual prompts. Our key idea is to leverage inter- and intra-class prototypes to capture class-specific knowledge and relationships. We propose two main components: (1) a plug-and-play contextual modulation module that integrates multi-scale information, and (2) a class-guided cross-attention mechanism that fuses prototypes and features for automatic prompt generation. Experiments on a public multi-organ dataset and a private ventricle dataset demonstrate that PGP-SAM achieves superior mean Dice scores compared with existing prompt-free SAM variants, while using only 10\% of the 2D slices.

## Usage
A suitable conda environment named `pgp-sam` can be created and activated with:

```bash
conda create -n pgp-sam python==3.10 -y
conda activate pgp-sam
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install -r requirements.txt
```

## Training
### SABS
```bash
python train.py --gpu_ids 0 --num_classes 9 --dataset sabs_sammed --scale 0.1 --model_type lora --batch_size 12 --num_workers 12 --loss ce --lr 3e-3 --dice_weight 0.8
```

### BHX
```bash
python train.py --gpu_ids 0 --num_classes 5 --dataset bhx_sammed --scale 0.05 --model_type lora --batch_size 16 --num_workers 16 --loss ce --lr 1e-3 --dice_weight 0.8
```

## Testing
### SABS
```bash
python infer.py --gpu_ids 0 --num_classes 9 --dataset sabs_sammed --scale 0.1 --model_type lora --train_time 20241030-1848 --volume True
python infer.py --gpu_ids 0 --num_classes 9 --dataset sabs_sammed --scale 0.1 --model_type lora --train_time 20241030-1848
```

### BHX
```bash
python infer.py --gpu_ids 0 --num_classes 5 --dataset bhx_sammed --scale 0.05 --model_type lora --train_time 20241029-1533 --volume True
python infer.py --gpu_ids 0 --num_classes 5 --dataset bhx_sammed --scale 0.05 --model_type lora --train_time 20241029-1533
```
