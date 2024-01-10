# Engilish
*  **Theory** : [https://wikidocs.net/225899](https://wikidocs.net/225899) <br>
*  **Implementation** : [https://wikidocs.net/226043](https://wikidocs.net/226043)

# 한글
*  **Theory** : [https://wikidocs.net/225899](https://wikidocs.net/225899) <br>
*  **Implementation** : [https://wikidocs.net/226043](https://wikidocs.net/226043)

This repository is folked from [https://github.com/yjh0410/RT-ODLab](https://github.com/yjh0410/RT-ODLab).
At this repository, simplification and explanation and will be tested at Colab Environment.

# YOLOX:

|   Model | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|---------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOX-S | 8xb8  |  640  |         40.1           |       60.3        |   26.8            |   8.9              | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolox_s_coco.pth) |
| YOLOX-M | 8xb8  |  640  |         46.2           |       66.0        |   74.3            |   25.4             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolox_m_coco.pth) |
| YOLOX-L | 8xb8  |  640  |         48.7           |       68.0        |   155.4           |   54.2             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolox_l_coco.pth) |
| YOLOX-X | 8xb8  |  640  |                        |                   |                   |                    |  |

- For training, we train YOLOX series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation.
- For optimizer, we use SGD with weight decay 0.0005 and base per image lr 0.01 / 64,.
- For learning rate scheduler, we use Cosine decay scheduler.

On the other hand, we are trying to use **AdamW** to train our reproduced YOLOX. We will update the new results as soon as possible.

|   Model | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|---------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOX-N | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOX-T | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOX-S | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOX-M | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOX-L | 8xb16 |  640  |                        |                   |                   |                    |  |
| YOLOX-X | 8xb16 |  640  |                        |                   |                   |                    |  |

- For training, we train YOLOX series with 300 epochs on COCO.
- For data augmentation, we use the large scale jitter (LSJ), Mosaic augmentation and Mixup augmentation.
- For optimizer, we use AdamW with weight decay 0.05 and base per image lr 0.001 / 64,.
- For learning rate scheduler, we use linear decay scheduler.

## Step 1. Clone from Github and install library

Git clone to root directory. 

```Shell
# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo-ML/Bible_4_Part_F_09_Pytorch_Yolox.git
# ! git pull origin master
! git pull origin main
```

A tool to count the FLOPs of PyTorch model.

```
from IPython.display import clear_output
clear_output()
```

```Shell
! pip install thop
```

## Step x. Download pretrained weight

```Shell
! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolox_s_coco.pth
! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolox_m_coco.pth
! wget https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolox_l_coco.pth
```

## Demo
### Detect with Image
```Shell
# Detect with Image

# See /content/det_results/demos/image
! python demo.py --mode image \
                 --path_to_img /content/dataset/demo/images/ \
                 --cuda \
                 -m yolox_s \
                 --weight /content/yolox_s_coco.pth \
                 -size 640 \
                 -vt 0.4
                 # --show
```

### Detect with Video
```Shell
# Detect with Video

# See /content/det_results/demos/video Download and check the results
! python demo.py --mode video \
                 --path_to_vid /content/dataset/demo/videos/street.mp4 \
                 --cuda \
                 -m yolox_s \
                 --weight /content/yolox_s_coco.pth \
                 -size 640 \
                 -vt 0.4 \
                 --gif
                 # --show
```

### Detect with Camera
```Shell
# Detect with Camera
# it don't work at Colab. Use laptop

# ! python demo.py --mode camera \
#                  --cuda \
#                  -m yolox_s \
#                  --weight /content/yolox_s_coco.pth \
#                  -size 640 \
#                  -vt 0.4 \
#                  --gif
                 # --show

```

## Download COCO Dataset

```Shell
# COCO dataset download and extract

# ! wget http://images.cocodataset.org/zips/train2017.zip
! wget http://images.cocodataset.org/zips/val2017.zip
! wget http://images.cocodataset.org/zips/test2017.zip
# ! wget http://images.cocodataset.org/zips/unlabeled2017.zip

# ! unzip train2017.zip  -d dataset/COCO
! unzip val2017.zip  -d dataset/COCO
! unzip test2017.zip  -d dataset/COCO

# ! unzip unlabeled2017.zip -d dataset/COCO

# ! rm train2017.zip
# ! rm val2017.zip
# ! rm test2017.zip
# ! rm unlabeled2017.zip

! wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
# wget http://images.cocodataset.org/annotations/image_info_test2017.zip
# wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

! unzip annotations_trainval2017.zip -d dataset/COCO
# ! unzip stuff_annotations_trainval2017.zip
# ! unzip image_info_test2017.zip
# ! unzip image_info_unlabeled2017.zip

# ! rm annotations_trainval2017.zip
# ! rm stuff_annotations_trainval2017.zip
# ! rm image_info_test2017.zip
# ! rm image_info_unlabeled2017.zip

clear_output()
```

## Test YOLOX
Taking testing YOLOX-S on COCO-val as the example,
```Shell
# Test Yolox

# See /content/det_results/coco/yolov1
! python test.py --cuda \
                 -d coco \
                 --data_path /content/dataset \
                 -m yolox_s \
                 --weight /content/yolox_s_coco.pth \
                 -size 640 \
                 -vt 0.4
                 # --show
```

## Evaluate YOLOX
Taking evaluating YOLOX-S on COCO-val as the example,
```Shell
# Evaluate Yolox

! python eval.py --cuda \
                 -d coco-val \
                 --data_path /content/dataset \
                 --weight /content/yolox_s_coco.pth \
                 -m yolox_s
```

# Training test
## Download VOC Dataset

```Shell
# VOC 2012 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
!tar -xvf "/content/VOCtrainval_11-May-2012.tar" -C "/content/dataset"
clear_output()

# VOC 2007 Dataset Download and extract

! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
! wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
!tar -xvf "/content/VOCtrainval_06-Nov-2007.tar" -C "/content/dataset"
!tar -xvf "/content/VOCtest_06-Nov-2007.tar" -C "/content/dataset"
clear_output()
```


## Train YOLOX
### Single GPU
Taking training YOLOX-n on COCO as the example,
```Shell
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolox_n \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
# yolox_n , yolox_t , yolox_s , yolox_m. yolox_l, yolox_x
```

```
# Cannot train yolox_t
# ! python train.py --cuda \
#                   -d voc \
#                   --data_path /content/dataset \
#                   -m yolox_t \
#                   -bs 16 \
#                   --max_epoch 5 \
#                   --wp_epoch 1 \
#                   --eval_epoch 5 \
#                   --fp16 \
#                   --ema \
#                   --multi_scale
```

```
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolox_s \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

```
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolox_m \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

```
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolox_l \
                  -bs 16 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

```
# GPU memory 12.3G at T4
! python train.py --cuda \
                  -d voc \
                  --data_path /content/dataset \
                  -m yolox_x \
                  -bs 8 \
                  --max_epoch 5 \
                  --wp_epoch 1 \
                  --eval_epoch 5 \
                  --fp16 \
                  --ema \
                  --multi_scale
```

### Multi GPU
Taking training YOLOX-S on COCO as the example,
```Shell
# Cannot test at Colab-Pro + environment

# ! python -m torch.distributed.run --nproc_per_node=8 train.py \
#                                   --cuda \
#                                   -dist \
#                                   -d voc \
#                                   --data_path /content/dataset \
#                                   -m yolox_s \
#                                   -bs 128 \
#                                   -size 640 \
#                                   --wp_epoch 3 \
#                                   --max_epoch 300 \
#                                   --eval_epoch 10 \
#                                   --no_aug_epoch 20 \
#                                   --ema \
#                                   --fp16 \
#                                   --sybn \
#                                   --multi_scale \
#                                   --save_folder weights/
```

