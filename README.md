# LMAffordance3D
Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions

## News
[2025-03] The paper is accepted by CVPR 2025.

## Abstract
Grounding 3D object affordance is a task that locates objects in 3D space where they can be manipulated, which links perception and action for embodied intelligence. For example, for an intelligent robot, it is necessary to accurately ground the affordance of an object and grasp it according to human instructions. In this paper, we introduce a novel task that grounds 3D object affordance based on language instructions, visual observations and interactions, which is inspired by cognitive science. We collect an Affordance Grounding dataset with Points, Images and Language instructions (AGPIL) to support the proposed task. In the 3D physical world, due to observation orientation, object rotation, or spatial occlusion, we can only get a partial observation of the object. So this dataset includes affordance estimations of objects from full-view, partial-view, and rotation-view perspectives. To accomplish this task, we propose LMAffordance3D, the first multi-modal, language-guided 3D affordance grounding network, which applies a vision-language model to fuse 2D and 3D spatial features with semantic features. Comprehensive experiments on AGPIL demonstrate the effectiveness and superiority of our method on this task, even in unseen experimental settings. 
<!--
## Dataset
Download data at [BaiduCloud]() or [Google Drive]() and organize as follows:
```
data
└-── Full_view
│    ├── Seen
│    │   ├── Description
│    │   ├── Img
│    │   └── Point
│    └── Unseen
│        ├── Description
│        ├── Img
│        └── Point
└-── Partial_view
│    ├── Seen
│    │   ├── Description
│    │   ├── Img
│    │   └── Point
│    └── Unseen
│        ├── Description
│        ├── Img
│        └── Point
└-── Rotation_view
     ├── Seen
     │   ├── Description
     │   ├── Img
     │   └── Point
     └── Unseen
         ├── Description
         ├── Img
         └── Point
```
## Install
```
conda create -n lmaffordance3d python=3.8 -y
conda activate otvic
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install requirements.txt -r
```

## Run
To train and evaluate LMAffordance3D model, run the following command:
```python
cd LAVIS
python train.py --cfg-path xxx.yaml
# eg:
# python train.py --cfg-path ./lavis/projects/affordance/LMAffordance3D/Full_view/Seen.yaml
# python train.py --cfg-path ./lavis/projects/affordance/LMAffordance3D/Full_view/Uneen.yaml
# python train.py --cfg-path ./lavis/projects/affordance/LMAffordance3D/Partial_view/Seen.yaml
# python train.py --cfg-path ./lavis/projects/affordance/LMAffordance3D/Partial_view/Uneen.yaml
# python train.py --cfg-path ./lavis/projects/affordance/LMAffordance3D/Rotation_view/Seen.yaml
# python train.py --cfg-path ./lavis/projects/affordance/LMAffordance3D/Rotation_view/Uneen.yaml
```
For visualization, you can modify the path of image and point, and then run ```visualization.py```.


## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
```

## Acknowledgement
Many thanks to these excellent open source projects:
 - [LAVIS](https://github.com/salesforce/LAVIS)
 - [IAGNet](https://github.com/yyvhang/IAGNet)

-->
## Other
Coming soon!
