# Union-Redefined Prototype Network for Scene Graph Generation

## Overview
![UP-Net Architecture](https://github.com/user-attachments/assets/46f49e79-30df-4c9a-b7f2-c3c99afe6f2e)

## Abstract

Recent advances in scene graph generation employ commonsense knowledge to model visual prototypes for various predicates. However, indiscriminate application of prototypes to all entity pairs within a predicate can lead to confusion when interpreting the same entity pairs that share similar visual features but correspond to different predicates. Particularly, clearly distinguishing predicates often requires careful consideration of subtle factors, such as gaze direction, spatial distance, and surrounding context. While these aspects are inherently part of the non-overlapping regions between the subject and object, they are frequently overlooked in practice. In this paper, we propose the Union-Redefined Prototype Network (UP-Net), which effectively captures predicate-specific visual nuances by leveraging the subject and object’s positional information and non-overlapping areas. Our method redefines entity pairs by incorporating the often-overlooked union region, excluding the intersection between the subject and object. Furthermore, we enhance the discriminative power among predicates by increasing the divergence between predicate-specific representations of the same entity pairs, thereby capturing the subtle visual nuances associated with each predicate. Extensive experiments on the Visual Genome dataset demonstrate that our approach achieves state-of-the-art performance.

---

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset
Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Train
```
python3 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR UP-Net \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 1024 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
  SOLVER.CHECKPOINT_PERIOD 30000 GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ./checkpoints/UP-Net_SGDet \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0;
```

## Device
All our experiments are conducted on one NVIDIA GeForce RTX 4090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Acknowledgement
The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Citation
If you use this work, please cite:
```bibtex
@article{Jung2025,
  title = {Union-Redefined Prototype Network for scene graph generation},
  journal = {Expert Systems with Applications},
  volume = {280},
  pages = {127486},
  year = {2025},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2025.127486},
  url = {https://www.sciencedirect.com/science/article/pii/S095741742501108X},
  author = {NamGyu Jung and Chang Choi},
  keywords = {Scene graph generation, Visual relation detection, Predicate differentiation, Non-overlapping regions, Commonsense knowledge}
}
