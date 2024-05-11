# Fine-tuning Segment Anything (SAM)

## Info

> This document was translated into English using [Claude](https://claude.ai/). 

This repository supports **fine-tuning** the **Segment Anything Model (SAM)**. By default, it trains the mask decoder of SAM on a given dataset. 

The training code mimics the operational process of the SamPredictor class from the [official SAM implementation](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py). 

This repository supports **batch sizes $\ge$ 2** and **enables multi-GPU training** based on PyTorch's DistributedDataParallel. Since the code is written with multi-GPU usage in mind, there may be a slight performance degradation when running on a single GPU. 

Additionally, **TinyViT**, proposed in [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), can be used as the image encoder by providing the appropriate model type. 

The supported model types are as follows, where vit_t refers to TinyViT:
- vit_h
- vit_l
- vit_b
- vit_t

Since this repository was created for the need of segmentation on a single class and background, it can **only handle binary-class datasets** by default. That is, the input **GT mask y must have values of either 0 or 1**.

The input prompt for the mask decoder is automatically generated. **By default, a rectangle(box) is created** using cv2.boundingRect that completely contains the binary mask region. 

If you want to apply it to a multi-class dataset, modify ```segment_anything/utils/trainer.py```. While I haven't implemented it myself, you can obtain multiple binary masks by one-hot encoding the GT mask y, and then obtain a box prompt for each mask. 

If the validation score improves, it saves the model weights. However, saving the entire set of weights is inefficient, so it **only saves the parameters being trained**. For more details, refer to ```segment_anything/utils/save_weight.py```. After loading the model with an existing checkpoint, you can load a part of the weights using ```load_partial_weight```. An example is in ```load_model_ex.py```

## How to use 

> To run the code, you need libraries such as ```numpy```, ```torch```, ```torchinfo```, and ```timm```(for vit_t).

- [Download the checkpoint](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) corresponding to the model type. 
- Inside ```run.py```, provide the appropriate paths for the train and validation datasets. 
- You can check the meaning of each argument in argparse at the top of ```run.py```. 
- For example, you can run like this: ```python run.py --batch_size 8 --port 1234 --dist True --seed 21 --model_type vit_h --checkpoint sam_vit_h.pth --epoch 5 --lr 2e-4```

The number of num_workers is set to 4 times the number of available GPUs. If you want to change this, modify opts.num_workers at the bottom of ```run.py```.

For model training, BCELoss and IoULoss were used, and IoU and Dice were used as evaluation metrics. You can find the detailed implementation in ```segment_anything/utils```. 

If you want to use different losses or evaluation metrics, modify ```run.py``` and ```segment_anything/utils/trainer.py```.
