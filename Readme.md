SwinMin for Mineral Images Recognition
Description
This is the repository for the code in Jia et al 2023. We design a SwinMin model for mineral photo image recognition in this paper, which embeds convolution information into the Transformer sequences and fuses multi-scale features with the proposed dynamic feature fusion module to exploit multi-scale contexts more effectively. SwinMin is based on Swin Transformer (https://arxiv.org/pdf/2103.14030.pdf), and we borrow the code from its repository (https://github.com/microsoft/Swin-Transformer). can be used to easily to recognize 45 categories mineral.
Requirement
torch
timm
numpy
Usage
you can run this code to train SwinMin in your dataset:
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --resume  your-path-of-the-pretrained-Swin-Tiny --data-path your-path-of-the-dataset
Pretrained model in mineral dataset
|name	| resolution |	acc@1 |	acc@5 |	#params	| model |
| ----- | ---------- | ------ | ----- | ------- | ----- |
| SwinMin |	224*224	| 92.86% |	98.75% |	32.67M |	link |
Dataset
── dataset_name                   
|   ├── train
|   |   ├── class_1
|   |   |	   ├── 1_1_images
|   |   |	   ├── 1_2_images
|   |   |	   ├── .....
|   |   ├── class_2
|   |   |	   ├── 2_1_images
|   |   |	   ├── 2_2_images
|   |   |	   ├── .....
|   |   ├── .....
|   |   ├── class_X
|   ├── val
|   |   ├── class_1
|   |   |	   ├── 1_1_images
|   |   |      ├── 1_2_images
|   |   |	   ├── .....
|   |   ├── class_2
|   |   |	   ├── 2_1_images
|   |   |	   ├── 2_2_images
|   |   |	   ├── .....
|   |	├── .....
|   |   ├── class_X
