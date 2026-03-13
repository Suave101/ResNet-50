# ResNet-20 Implementation

## 📌 Overview

This repository contains an implementation of the **ResNet-20 (Residual Network)** architecture, a milestone in deep learning for computer vision. This project was developed as part of my portfolio to demonstrate my understanding of deep Convolutional Neural Networks (CNNs), specifically focusing on solving the vanishing gradient problem using skip connections.

The structural foundation and coding approach for this implementation were guided by the comprehensive tutorials provided by **GeeksforGeeks**, adapted and optimized for my specific dataset and training pipeline.

## 🚀 Key Features

* **Custom Implementation:** Built ResNet-20 from the ground up using `[TensorFlow / Keras]`.
* **Simple Architecture:** Correctly implemented the 3x3 convolutional simple blocks.
* **Skip Connections:** Integrated identity and projection shortcuts to ensure smooth gradient flow across 20 layers.
* **Custom Training Loop:** Includes modular scripts for training, validation, and evaluation.

## 🏗️ Architecture Highlights

ResNet-20 utilizes **Residual Blocks** to allow the training of deep networks. Instead of hoping each few stacked layers directly fit a desired underlying mapping, these blocks are explicitly allowed to fit a residual mapping.

* **Conv1:** 7x7 convolution, 64 filters, stride 2, followed by max pooling.
* **Output:** Global Average Pooling followed by a Fully Connected layer for classification.

## 📂 Repository Structure

```text
├── data/                   # Directory for dataset (not tracked by git)
├── models/
│   └── resnet20.py         # The core ResNet-20 architecture implementation
├── utils/
│   ├── dataset.py          # Data loaders and augmentation pipelines
│   └── helper.py           # Helper functions for plotting and logging
├── train.py                # Main script to train the model
├── test.py                 # Script to evaluate model performance
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation

```

## 🛠️ Installation & Usage

**1. Clone the repository:**

```bash
git clone https://github.com/Suave101/ResNet-20.git
cd ResNet-20

```

**2. Install dependencies:**

```bash
pip install -r requirements.txt

```

**3. Run Training:**

```bash
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001

```

**4. Evaluate the Model:**

```bash
python test.py --weights_path checkpoints/best_model.pth

```

## 📊 Dataset & Results

* **Dataset:** Trained and evaluated on the `[CIFAR-10]`.
* **Performance:** Achieved an accuracy of **[XX.X]%** on the test set.

## 💡 Learnings & Takeaways

Through building this project, I solidified my understanding of:

* How degradation problems occur in deep networks and how residual learning mitigates them.
* Translating academic papers and technical articles (like those on GeeksforGeeks) into functional, clean code.
* Managing dimensions and tensor shapes across complex bottleneck structures.

## 👤 Author

**Alexander Doyle**

* [LinkedIn](https://www.linkedin.com/in/alexander-doyle-a7a790245/)
