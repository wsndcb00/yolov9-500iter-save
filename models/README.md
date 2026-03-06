# Models Directory

## Overview

This directory is intended for storing **custom model architectures, pretrained weights, and model configuration files** related to this project.

In the current version of this repository, the model architecture is **not implemented directly inside this directory**. Instead, the model is provided by the **Ultralytics YOLO framework**, which dynamically loads the model structure and pretrained weights during training.

The training script utilizes the **YOLOv9c** pretrained model as the initialization backbone.

---

## Model Used in This Project

The training pipeline loads the pretrained model:

```
yolov9c.pt
```

This model is automatically downloaded by the Ultralytics framework if it is not found locally.

The model is then fine-tuned on the custom dataset during training.

Example from the training script:

```python
from ultralytics import YOLO

model = YOLO("yolov9c.pt")
```

---

## Possible Future Extensions

This directory is reserved for future model-related extensions, such as:

* Custom YOLOv9 architecture modifications
* Alternative backbone networks
* Custom detection heads
* Lightweight or optimized model variants
* Experiment-specific pretrained weights

Examples of files that may appear here in the future:

```
models/
│
├── custom_yolov9.yaml
├── custom_backbone.py
├── optimized_head.py
└── pretrained_weights/
```

---

## Why This Directory Exists

In many **machine learning and computer vision repositories**, the `models/` directory is used to organize all model-related components separately from training scripts and utilities.

Keeping model definitions here helps maintain a **clean and modular project structure**, which is especially important for research projects and reproducible experiments.

---

## Note

For this repository, the core modification focuses on **improving the checkpoint saving strategy during training**, where model checkpoints are saved **every 500 iterations instead of once per epoch**.

The model architecture itself remains unchanged from the original YOLOv9 implementation.
