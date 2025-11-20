# deeplearning

# Deep Learning & Image Classification Experiments

This repository contains a collection of Jupyter Notebooks implementing various Deep Learning architectures using **TensorFlow** and **Keras**. The projects explore Convolutional Neural Networks (CNNs), Artificial Neural Networks (ANNs), and hyperparameter tuning across different datasets (MNIST, Cats vs. Dogs, and a custom Face dataset).

## 📂 Repository Contents

### 1. `MNIST_LeNet_vs_ANN.ipynb`
**Objective:** A comparative analysis between a classic Convolutional Network (LeNet-5) and a Standard Artificial Neural Network (MLP) on the MNIST handwritten digit dataset.

* **Models Implemented:**
    * **LeNet-5:** Implements the classic 1998 architecture using Tanh activations, Average Pooling, and 32x32 input padding.
    * **Standard ANN:** A dense Multi-Layer Perceptron treating images as flattened 784-pixel vectors.
* **Key Findings:**
    * The LeNet-5 model (~98-99% accuracy) outperforms the ANN (~97%) by leveraging spatial hierarchies via convolutions.
    * Includes visualization of Training Time vs. Accuracy.

### 2. `cat_vs_dog.ipynb`
**Objective:** Binary image classification using a custom CNN on a restricted subset of the "Cats vs. Dogs" dataset.

* **Methodology:**
    * **Data Pipeline:** Uses `tensorflow_datasets` with image resizing (160x160), normalization, and batching.
    * **Experiment:** Intentionally limits the training data to **250 cats and 250 dogs** (500 total) to observe model behavior under data scarcity.
    * **Architecture:** A sequential CNN with 3 Convolutional blocks, Max Pooling, and Dropout for regularization.
* **Outcome:** Demonstrates the challenges of training deep networks on small datasets (achieving approx. 58-60% accuracy without transfer learning).

### 3. `Face_Recognition_Experiments.ipynb`
**Objective:** Hyperparameter tuning and architectural experimentation for a Face Recognition task using a custom dataset.

* **Experiments Conducted:**
    1.  **Network Depth:** Comparing performance between 1, 2, and 3 Convolutional layers.
    2.  **Regularization:** Analyzing the impact of **Dropout** layers on validation accuracy.
    3.  **Input Resolution:** Benchmarking accuracy and training time across image sizes: **32x32, 64x64, and 128x128**.
* **Key Insights:**
    * Identified that 64x64 or 128x128 resolution provides the best balance for face features.
    * Dropout significantly improved generalization (reducing overfitting).

## 🛠️ Tech Stack
* **Python 3.x**
* **TensorFlow / Keras** (Deep Learning API)
* **NumPy** (Matrix operations)
* **Matplotlib** (Visualization)
* **TensorFlow Datasets** (Data loading)

## 🚀 Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
