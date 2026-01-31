# Module 4: Applied Deep Learning with PyTorch ⚡

**From PyTorch Fundamentals to CNNs and Transfer Learning**

**📍 Location:** `4_PyTorch/`
**🎯 Prerequisite:** Module 3 – Neural Networks from Scratch
**➡️ Next Module:** Advanced Deep Learning Architectures

Welcome to **Module 4** of **SAIR**, focused on **applied deep learning using PyTorch**.
This module bridges the gap between theory and practice by introducing PyTorch as a full deep learning framework and progressively building toward **convolutional neural networks (CNNs)** and **transfer learning with modern architectures**.

You will move from core PyTorch concepts to training real models on real datasets using industry-standard workflows.

---

## 🎯 Who Is This Module For?

### ✅ This module is suitable if you:

* Understand basic neural networks and backpropagation
* Want to use PyTorch for practical deep learning projects
* Need hands-on experience with CNNs and transfer learning
* Are preparing for applied ML, CV, or research-oriented roles

### 🔁 You may skim or review if you already:

* Have experience training CNNs in PyTorch
* Understand DataLoader optimization and GPU workflows
* Have implemented transfer learning with pretrained models

---

## 🛠️ Core Technologies

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge\&logo=nvidia\&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge\&logo=jupyter\&logoColor=white)

</div>

---

## 📚 Module Contents

### **Notebooks**

| File                              | Focus                                                           |
| --------------------------------- | --------------------------------------------------------------- |
| **`1_Intro.ipynb`**               | PyTorch fundamentals: tensors, autograd, models, training loops |
| **`2_DataLoader.ipynb`**          | Dataset & DataLoader design, performance considerations         |
| **`3_CNN.ipynb`**                 | Convolutional Neural Networks from scratch                      |
| **`4_Transfer_and_ResNet.ipynb`** | Transfer learning, ResNet, and pretrained models                |

---

### **Labs**

| Lab               | Description                        |
| ----------------- | ---------------------------------- |
| **`lab_1.ipynb`** | PyTorch basics & tensor operations |
| **`lab_2.ipynb`** | Training neural networks           |
| **`lab_3.ipynb`** | CNN implementation and experiments |
| **`lab_4.ipynb`** | Transfer learning & evaluation     |

Student submissions are organized under:

```
lab_assignments/
├── student_name/
│   └── lab_1.ipynb
```

---

### **Data**

```
data/
└── cifar-10-batches-py/
```

* CIFAR-10 dataset for CNN and transfer learning experiments
* Includes raw batch files and metadata
* Used across CNN and transfer learning notebooks

---

### **Reference Papers**

| File                    | Purpose                       |
| ----------------------- | ----------------------------- |
| **`AlexNet_paper.pdf`** | Foundational CNN architecture |
| **`ResNet_paper.pdf`**  | Deep residual learning        |

These papers provide architectural context for the models implemented in the notebooks.

---

### **Assets**

```
assets/
└── ME.jpeg
```

Used for demonstrations, visualization, or documentation examples.

---

## 🗺️ Learning Progression

### **Phase 1: PyTorch Foundations**

📘 `1_Intro.ipynb`

* Tensors and tensor operations
* Automatic differentiation (autograd)
* Building models with `nn.Module`
* Training loops and evaluation
* CPU/GPU device handling
* Saving and loading models

---

### **Phase 2: Data Pipelines**

📦 `2_DataLoader.ipynb`

* Custom `Dataset` classes
* Efficient `DataLoader` usage
* Batching, shuffling, workers
* Common data pipeline pitfalls

---

### **Phase 3: Convolutional Neural Networks**

🧠 `3_CNN.ipynb`

* Convolutions, pooling, padding
* CNN architecture design
* Training CNNs on CIFAR-10
* Overfitting, regularization, diagnostics

---

### **Phase 4: Transfer Learning & Modern Architectures**

🚀 `4_Transfer_and_ResNet.ipynb`

* Motivation for transfer learning
* Feature extraction vs fine-tuning
* Using pretrained ResNet models
* Performance comparison and analysis

---

## 📂 Directory Structure

```
4_PyTorch/
│
├── README.md
├── 1_Intro.ipynb
├── 2_DataLoader.ipynb
├── 3_CNN.ipynb
├── 4_Transfer_and_ResNet.ipynb
│
├── labs/
│   ├── lab_1.ipynb
│   ├── lab_2.ipynb
│   ├── lab_3.ipynb
│   └── lab_4.ipynb
│
├── lab_assignments/
│   └── student_submissions/
│
├── data/
│   └── cifar-10-batches-py/
│
├── assets/
│
├── AlexNet_paper.pdf
└── ResNet_paper.pdf
```

---

## 🎯 Learning Outcomes

After completing this module, you should be able to:

* Use PyTorch tensors and autograd confidently
* Implement neural networks using `nn.Module`
* Write clean and correct training loops
* Build and train CNNs for image classification
* Load and preprocess data efficiently
* Apply transfer learning with pretrained models
* Understand key CNN architectures from literature
* Evaluate and compare deep learning models

---

## 🚀 How to Get Started

```bash
# Start with fundamentals
jupyter notebook 1_Intro.ipynb

# Learn data pipelines
jupyter notebook 2_DataLoader.ipynb

# Build CNNs
jupyter notebook 3_CNN.ipynb

# Apply transfer learning
jupyter notebook 4_Transfer_and_ResNet.ipynb

# Practice with labs
cd labs
```

---

## 📌 Notes

* Labs are meant for **active practice**
* Papers are provided for **conceptual understanding**
* CIFAR-10 is used consistently for reproducible experiments
* Code emphasizes clarity and correctness over shortcuts

---

## 🔜 What’s Next?

After this module, you’ll be ready to:

* Explore advanced architectures (Transformers, Vision Transformers)
* Optimize models and training pipelines
* Read and implement modern research papers
* Build end-to-end deep learning projects

➡️ **Next Module:** *Advanced Deep Learning Architectures*

---

> **“Frameworks automate computation — understanding gives you control.”**

This module is about **learning PyTorch deeply enough to use it correctly, confidently, and creatively**.
