# Đề tài

Topic: **Plant Leaf Disease Detection**   
Problem Formulation:

Input Image \-\> Data Augmentation,Preprocessing → CNN/CV Architecture → Output disease with bounding box   
![][image1]  
Targets:

* **Technical Aspect:** Build a complete pipeline for a lightweight Deep Learning model that achieves high classification accuracy on large datasets in complex environments.  
* **Application Aspect**: Automate the crop health monitoring process, enabling early disease detection (when it is hard to see with the naked eye), thereby helping farmers reduce pesticide waste, protect the environment, and increase productivity.

Focusing on:  
•Real-time processing capability  
•Faster inference speed  
•Suitable for agricultural deployment

### 

## Models

Lightweight Classification Networks (for Mobile/IoT):  
•MobileNet (V1, V2, V3): Uses depthwise separable convolutions; optimized for low-latency mobile inference.  
•EfficientNet (B0 to B7): Automatically balances network depth, width, and resolution; delivers extreme accuracy with very few parameters.  
Object Detection Models (Disease Region Localization):  
•One-stage (Fast, Real-time): YOLO (V3, V4, V5, V8) and SSD. Ideal for real-time detection of multiple lesions on a single leaf.  
•Two-stage (High Accuracy): Faster R-CNN or Mask R-CNN (capable of pixel-perfect lesion segmentation).  
•Anchor-free Detectors: Systems like CenterNet or RT-DETR that simplify the detection process by removing the need for predefined anchor boxes.

## Benchmark Datasets for Lightweight, Real‑World Applications

General:  
•PlantDoc: 2,598 leaf images in natural, noisy environments; ideal for testing lightweight models under occlusion and background clutter.  
•IDADP: Agricultural disease/pest images captured under complex field conditions, suited for edge deployment.  
Specific Plants:  
•Potato Leaf (Healthy/Late Blight): 426 farm images in noisy settings; efficient for rapid training/testing.  
•Cucumber Plant Diseases: 695 field images under natural conditions; optimized for lightweight models.  
•BananaLSD: 937 smartphone‑captured banana leaf spot images in diverse real‑world conditions.  
•Corn (CD\&S): 4,455 handheld field images with realistic background complexity.  
•RoCoLe (Robusta Coffee Leaf): 1,560 smartphone images in uncontrolled environments, capturing varied lighting.  
•BRACOL (Arabica Coffee): 1,747 images from five mobile devices, reflecting variability in camera quality.

## Expected Results

•A Fully Functional Pipeline: The successful development of a complete pipeline utilizing a lightweight Deep Learning model.  
•High Accuracy in Complex Environments: Achieving high classification accuracy on large datasets, even when deployed in complex, real-world environments. As a benchmark, top CNN models are expected to exceed 99% accuracy.  
•Optimized for Mobile/IoT: The final model must demonstrate real-time processing capabilities and fast inference speeds. This means achieving favorable operational metrics like high FPS (Frames Per Second), low latency, and low model complexity (reduced parameters and FLOPs) suitable for edge deployment.

# Literature Review

This project builds on several state-of-the-art literature sources reviewing deep learning algorithms for plant disease recognition:
- **Harakannanavar et al. (2022)** & **Salka et al. (2025)**: Synthesize standard CNN pipelines (HE, K-means segmentation, GLCM/DWT/PCA feature extraction, and SVM/KNN/CNN classification).
- **Zhao et al. (2025)**: Classify object detectors in agriculture into two-stage (Faster R-CNN), one-stage (YOLO, SSD), and anchor-free (CenterNet, RT-DETR) categories.
- **Sujatha et al. (2025)**: Benchmark hybrid CNN + ML models (using backbones like VGG19 or Inception v3 as feature generators for SVM/RF/KNN) and highlight crop-dependent optimal combinations.

# Data Survey

We focus on two main datasets:
1. **PlantVillage**: Lab-controlled environment, 54,305+ images, 38 classes, uniform backgrounds.
2. **PlantDoc**: Field-collected noisy environment, 2,598 images, 13 species, 17 classes, complex backgrounds.

# PlantDoc

## **1\. Handling Blur, Noise, and Illumination Variations (STEP 5 — Transforms)**

The implementation utilizes a custom transforms.Compose pipeline to address the sub-optimal quality of web-scraped images:

* **Handling Blur and Camera Artifacts:** This is addressed indirectly using transforms.RandomErasing(p=0.15, ...), which randomly masks out small rectangular patches of the image. This forces the deep learning architecture to learn local, resilient features of the disease rather than relying on global leaf geometry.  
* **Simulating In-Field Illumination Changes:** To counter varying exposure, shadows, or pre-processed web graphics, transforms.ColorJitter(0.3, 0.3, 0.2, 0.05) is applied. This randomly jitters brightness, contrast, saturation, and hue, preventing the model from over-fitting to specific lighting conditions.

## **2\. Image Standardization and Geometric Scaling**

* **Uniform Spatial Resolution:** The script enforces a standard resolution of **224x224 pixels** (IMG\_SIZE \= 224), ensuring structural compatibility with advanced backbones like ViT, Swin Transformer, ConvNeXt, and ResNet.  
* **Dynamic Cropping Strategy:** During the training phase, images are upscaled and cropped using a two-stage process:  
* Python

transforms.Resize((IMG\_SIZE+32, IMG\_SIZE+32)),  
transforms.RandomCrop(IMG\_SIZE),

*   
* Images are first scaled up to 256x256 pixels before a 224x224 patch is randomly extracted. This introduces spatial translation invariance, training the networks to detect symptoms regardless of whether the infected region is centered or positioned near the frame boundaries.

## **3\. Mitigating Label Noise and Corrupted Annotations (STEP 7\)**

Crowdsourced datasets like PlantDoc suffer heavily from label noise (e.g., misclassified pathological symptoms). The code counteracts this using two advanced regularization mechanisms:

* **Label Smoothing Regularization (LabelSmoothCE):** Instead of computing standard Cross-Entropy against hard, one-hot encoded targets (where true class \= 1.0, incorrect classes \= 0.0), a smoothing factor of LABEL\_SMOOTH \= 0.1 is introduced. The target probability for the ground-truth class drops to 0.9, while the remaining 0.1 is distributed evenly across the other categories. This dampens penalties during backpropagation if the network predicts against a corrupted label.  
* **Mixup Augmentation (mixup):** Pairs of random training samples are linearly interpolated by a factor of $\\lambda$ drawn from a Beta distribution (e.g., mixing $60\\%$ of a diseased tomato leaf with $40\\%$ of a healthy apple leaf). The corresponding labels are blended proportionally. This smooths out decision boundaries, drastically reducing the model's tendency to memorize mislabeled outliers.

## **4\. Resolving Severe Class Imbalance (FILTER 1 & 2, STEP 4\)**

This section contains the most comprehensive structural logic in the preprocessing pipeline, designed to prevent majority-class bias:

* **Syntactic Class Unification:** The script implements PLANTDOC\_MAP to act as a data-cleansing lookup table, mapping irregular, lower-case, or fractured folder names from PlantDoc into a rigorous, unified nomenclature (UNIFIED\_CLASSES).  
* **Inter-Dataset Validation Filter:** FILTER 1 & 2 drop any crop species that do not share both a healthy and at least one diseased state across both PlantDoc and PlantVillage datasets (\_dropped\_crops), filtering out dead-end categories that do not contribute to multi-crop disease discrimination.  
* **Over-Sampling via Augmentation Balancing:** The pipeline defines an exact training budget per class (N\_TRAIN \= 1000). For minority classes starved of real samples, an automated loop replicates available real images until the limit is hit:

Python  
 while len(out\_tr) \< nt:  
      out\_tr.append(rng.choice(tr\_real))  
      train\_is\_aug.append(True)

This creates a completely uniform, balanced categorical distribution for the training phase, preventing the model from gaming classification accuracy by over-predicting dominant classes.

## **5\. Elimination of Data Leakage**

* **Strict Split Isolation:** The over-sampling/replication mechanism assigns a tracking flag (train\_is\_aug \= True) and is strictly confined to the **Train Loader**.  
* The **Validation** and **Test** splits are constructed exclusively from unique, original assets (REAL images only — no padding). This guarantees that no identical or replicated validation targets leak into the evaluation pipeline, enforcing a statistically sound benchmark for model generalization.

# IDAPD

# **IDADP**

**Source**: [https://www.scidb.cn/en/detail?dataSetId=633694461276192770](https://www.scidb.cn/en/detail?dataSetId=633694461276192770)

1. **Stats**:  
- **Number**: 17 624 images  
- **Sub-datasets:** 15  
  - Rice: 6 sets  
  - Wheat: 5 sets  
  - Maize: 4 sets  
- **Image Conditions:**   
  - High resolution, consistent sizes  
  - Background: Mixed (Real-life image and laboratory conditions)  
  - Label: Each sub-datasets is a disease of a specific crop  
- **Problems:**  
  - Class Imbalance  
  - Missing Control Group  
  - Background Interference  
      
2. **Solutions:**

**Preprocessing and Cleaning:**

- Nearest-Neighbor Interpolation for Downscaling: Standard tensor size destroys texture of fugal networks.   
    
- RGB Channel Standardization: Researchers calculate mean and standard deviation for channel red, green, blue for the entire training subset, then subtract the mean for every tensor. This forces the model to learn features instead of brightness.  
    
  *Ref: [https://pmc.ncbi.nlm.nih.gov/articles/PMC8942633/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8942633/)*


**Object Detection:**

- Map IDAPD images to object detection architectures like YOLO to detect infectious sections, then process with classifiers.

	*Ref: [https://www.mdpi.com/2077-0472/13/7/1361](https://www.mdpi.com/2077-0472/13/7/1361)*

**Fine Tuning:**

- To combat the lack of healthy samples, researchers train models on balanced datasets like PlantVillage first. Then freeze the early feature-extraction layer. Finally, they fine tune on real-life noisy datasets like IDADP.

	**Data Augmentation:**

- [**PlantPAD**](https://academic.oup.com/nar/article/52/D1/D1556/7332078)**: Researchers aggregate multiple datasets to ensure coverage, alleviating the lack of healthy samples in IDADP.**  
    
3. **Standard Pipeline:**  
- Data Augmentation:   
  - Spatial: Rotate, flip, crop  
  - Color: Jitter brightness, contrast, saturation  
  - Adversarial: Random erasing

# Model Survey

# Depthwise CNN

**[Enhancing plant disease detection through deep learning: a Depthwise CNN with squeeze and excitation integration and residual skip connections](https://drive.google.com/file/d/1MTcFz9IDUnv2YCSIl0SYAmz85i-zUMqT/view?usp=sharing)**

**CNN-Based Models**

Convolutional Neural Networks (CNNs) are the foundational architecture for computer vision in agriculture. They use layered spatial operations to process images step-by-step, starting with basic edges and building up to complex disease patterns.

**Core Mechanistic Workflow of a CNN**

A standard CNN processes an input leaf image through a structured pipeline of operations:

\[Input Image\] ──\> \[Convolution Layer (Feature Extraction)\] ──\> \[Pooling Layer (Downsampling)\]

   	│

   	└───\> \[Residual Skip Connections (Gradient Preservation)\] ──\> \[Fully Connected Layer (Classification)\]

1. **Convolution:** Small, learnable pixel matrices called kernels slide (convolve) across the input image. At each step, the kernel multiplies its weights with the underlying leaf pixels to generate a **Feature Map**. Early layers capture primitive elements like edges, lines, and textures. Deeper layers combine these primitives to recognize complex shapes like rust spots, powdery mildew patches, or leaf wilting patterns.  
2. **Pooling:** This operation down-samples the generated feature maps to reduce their spatial dimensions. For instance, **Max Pooling** extracts only the maximum pixel value within a localized window (e.g., $2\\times2$). This process reduces total computational overhead and helps the network maintain **translation invariance**, allowing it to identify a disease lesion regardless of its exact pixel coordinates on the leaf.  
3. **Classification Process:** After several rounds of convolution and pooling, the final 3D feature maps are flattened into a 1D vector. This vector passes through a series of **Fully Connected (Dense) Layers**, concluding with an activation function (like Softmax) that outputs a probability score for each plant disease class.

**Evaluation of Standard CNN Architectures**

**ResNet (Residual Networks)**

* **Architecture Overview:** ResNet introduces **Residual Skip Connections** that allow feature inputs to bypass one or more convolutional layers entirely.  
* **Mechanism:** Instead of forcing layers to learn a completely new mapping from scratch, the skip connection adds the original input $x$ directly to the output of the convolutional block $F(x)$, optimizing the residual mapping $F(x) \+ x$. This design directly mitigates the **vanishing gradient problem**, enabling researchers to train much deeper networks without experiencing performance drops.  
* **Strengths:** Highly reliable feature extraction; easy to train at scale.  
* **Weaknesses:** Large parameter size and high memory footprint, making it slow on low-resource mobile platforms.

**DenseNet (Densely Connected Convolutional Networks)**

* **Architecture Overview:** DenseNet modifies the skip-connection paradigm by connecting every single layer directly to every subsequent layer in a feed-forward fashion.  
* **Mechanism:** Rather than combining features through mathematical addition, DenseNet concatenates the feature maps of all preceding layers as inputs for the next layer. This design ensures maximum feature reuse across the network.  
* **Strengths:** High parameter efficiency and continuous feature propagation throughout the network.  
* **Weaknesses:** High memory utilization during training due to the continuous concatenation of feature maps.

**EfficientNet**

* **Architecture Overview:** EfficientNet balances accuracy and speed by using a systematic optimization technique called **Compound Scaling**.  
* **Mechanism:** Instead of arbitrarily deep or wide networks, Compound Scaling uniformly scales network depth, width, and input image resolution using a fixed, mathematically optimized scaling coefficient.  
* **Strengths:** Excellent accuracy-to-parameter ratio.  
* **Weaknesses:** Complex optimization schedules; can still show high inference latency on older hardware without dedicated neural accelerators.

**Deep Dive: Advanced Lightweight CNN (Ashurov et al., 2025\)**

To address the computational constraints of deploying models on agricultural edge devices, Ashurov et al. developed an optimized, lightweight Depthwise CNN framework.

**Problem Solved**

Standard convolutional networks require significant computing power and memory, making them impractical for real-time deployment on low-cost microcontrollers, offline mobile applications, or agricultural drones.

**Architecture and Mechanism**

The model optimizes efficiency by combining three core structural mechanisms:

1. **Depthwise Separable Convolutions:** The model replaces standard 3D convolutions with a two-step operation:  
   * **Depthwise Convolution:** A single spatial kernel is applied to each input channel independently to extract spatial features.  
   * **Pointwise Convolution:** A $1\\times1$ kernel mixes these independent channel outputs together. This separation cuts computational complexity by roughly $80\\text{--}90\\%$ compared to standard convolutions.  
2. **Squeeze-and-Excitation (SE) Block:** This channel-wise attention module adaptively recalibrates feature responses:  
   * **Squeeze:** Global Average Pooling compresses 2D spatial feature maps into a 1D channel descriptor vector.  
   * **Excitation:** A compact bottleneck fully-connected layer computes non-linear dependencies between channels, generating a set of weights. These weights scale the original feature maps, forcing the model to focus on relevant disease features while ignoring background noise.  
3. **Residual Skip Connections:** These layers pass gradients directly through the lightweight blocks, ensuring stable training and preventing information loss across deep layers.

**Why It Performs Well**

By decoupling spatial and channel operations via depthwise separation, the network remains compact. Simultaneously, the Squeeze-and-Excitation blocks dynamically direct the model's focus to the most informative disease features, allowing it to maintain high classification accuracy despite its reduced parameter count.

**Performance and Comparative Results**

When tested against multi-species datasets, this modified architecture achieved an **Accuracy of 98.00%** and an **F1-Score of 98.20%**. It performed on par with or better than much heavier models like ResNet50 and Inception-V3, while using only a fraction of the computational power (FLOPs).

![][image2]

# Hybrid CNN \+ ViT

**[s40747-024-01764-x (1).pdf](https://drive.google.com/file/d/1n7qK52KzczIAZ0O3x6VGlEnXgsfsl8eb/view?usp=sharing)**

**Hybrid CNN \+ Vision Transformer Models**

While CNNs are highly effective at capturing local details (like texture and edges), their small convolutional kernels struggle to capture long-range global relationships across an entire image. Hybrid models combine CNNs and Vision Transformers (ViTs) to capture both local details and global context.

**Technical Components of the Hybrid Approach**

**CNN Feature Extraction (Local Scope)**

The hybrid system uses an initial CNN backbone (or an ensemble of multiple networks) as a localized feature extractor. As an image passes through these convolutional layers, the network preserves spatial hierarchies and extracts fine-grained details, such as localized fungal lesions or fine leaf veins.

**Vision Transformer (ViT) Encoder (Global Scope)**

Once the CNN extracts these localized feature maps, they are converted into sequential tokens and fed into a Vision Transformer encoder.

The transformer treats different areas of the leaf image like words in a sentence, calculating relationships across the entire image regardless of distance.

**The Self-Attention Mechanism**

The core component of the Vision Transformer is **Multi-Head Self-Attention**. It maps tokens using three learnable vectors: Queries ($Q$), Keys ($K$), and Values ($V$). The attention scores are calculated using the following mathematical formula:

$$\\text{Attention}(Q, K, V) \= \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d\_k}}\\right)V$$

This dot-product calculation measures the similarity between every patch pair. This allows the network to assess how a specific symptom on one side of a leaf relates to discoloration or structural changes on the opposite side.

**Analytical Case Study: Ensemble Hybrid Framework (Aboelenin et al., 2025\)**

**Problem Solved**

This research addresses the accuracy limitations of standalone models. CNNs often miss global contextual patterns due to their localized receptive fields, while standalone Vision Transformers require massive datasets to learn spatial relationships because they lack the built-in spatial assumptions (inductive biases) of CNNs.

**Architecture and Workflow**

Aboelenin et al. designed a multi-stage hybrid system:

\[Input Leaf Image\] ──\> \[Ensemble CNN Block (VGG16 \+ Inception-V3 \+ DenseNet20)\]

                             │

                             └───\> \[Spatial Feature Maps\] ──\> \[Tokenization & Patch Flattening\]

                                                                     │

                                                                     └───\> \[ViT Encoder (Self-Attention)\] ──\> \[Classification Output\]

1. **Ensemble Convolution Stage:** The input image is processed in parallel by three distinct CNN backbones: **VGG16**, **Inception-V3**, and **DenseNet20**. Their outputs are combined into a rich, multi-scale feature map.  
2. **Tokenization Stage:** These combined spatial feature maps are flattened into 2D patches, linearly projected into patch embeddings, and combined with learnable **Position Embeddings** to preserve spatial layout information.  
3. **Transformer Processing Stage:** These vectors pass through the Multi-Head Self-Attention layers of the ViT encoder, mapping global contextual relationships before generating a final disease classification.

**Why the Hybrid Model Achieves Higher Accuracy**

The architecture combines the strengths of both paradigms. The CNN ensemble handles localized feature extraction (edges, colors, spots), while the downstream Transformer encoder maps global relationships across the entire leaf. This dual-focus approach prevents the model from misclassifying healthy sections or being thrown off by field noise.

**Performance and Comparative Results**

The framework achieved exceptional classification results across challenging target datasets:

* **Apple Leaf Disease Dataset:** **99.24% Accuracy**  
* **Corn Leaf Disease Dataset:** **98.00% Accuracy**

It consistently outperformed standalone VGG16, Inception-V3, and basic ViT setups, establishing a new baseline for classification precision. However, this high accuracy comes at the cost of increased computational complexity and longer inference times due to running multiple large networks simultaneously.

![][image3]

# YOLO Detection

**[s41598-025-14021-z (1).pdf](https://drive.google.com/file/d/1uqUxPxPGj2Wkz-PTXFEsui9AS0K_ieNP/view?usp=sharing)**

**YOLO-Based Detection Models**

Standard classification networks can only identify whether an image contains a disease. Object detection models go a step further by simultaneously identifying, classifying, and locating multiple disease spots within an image using spatial coordinates.

**Image Classification vs. Object Detection**

* **Image Classification:** Evaluates the image as a whole to output a single categorical label (e.g., "Tomato Leaf Late Blight"). It does not identify where the disease is located or how many lesions are present.  
* **Object Detection:** Identifies individual disease spots by drawing tight **Bounding Boxes** around each lesion, providing exact pixel coordinates ($x, y$ center, width, height) alongside a classification label and confidence score. This capability allows farmers to measure the exact severity and spread of an infection.

**Deep Dive: YOLO-LeafNet (Kaur et al., 2025\)**

**Problem Solved**

In real-world fields, multiple diseases can infect a single plant simultaneously, often appearing as tiny, scattered spots across a leaf. Standard classifiers cannot isolate or count these individual infected zones. Furthermore, most object detection models are too computationally heavy to run in real time on field equipment.

**Architecture and Operational Workflow**

YOLO-LeafNet is built upon the single-stage **You Only Look Once (YOLO)** real-time object detection paradigm. Unlike multi-stage detectors that generate regional proposals before classifying them, YOLO-LeafNet processes the entire image in a single forward pass:

1. **Input Pre-Processing & Augmentation:** Input images pass through a data augmentation pipeline that applies transformations like **Mosaic Augmentation** (combining four random training images into one) and color jittering. This step helps the network learn to identify small, multi-scale objects in cluttered environments.  
2. **Feature Backbone & Neck:** A series of dense convolutional layers extracts feature maps at multiple scales, using specialized cross-stage partial connections to maintain high processing speeds.  
3. **Detection Head Prediction:** The network divides the image into a grid. For each grid cell, the model predicts bounding boxes, class probabilities, and confidence scores simultaneously using a single regression calculation.

**Why It Performs Well**

YOLO-LeafNet processes images in a single step, which keeps its computational demands low and enables real-time detection. The integration of multi-scale feature tracking and mosaic augmentation ensures the network can accurately spot small, early-stage disease lesions even against complex, messy background field noise.

**Performance and Results**

Tested across diverse datasets containing Grape, Bell Pepper, Corn, and Potato leaves, YOLO-LeafNet achieved outstanding results:

* **mAP50 (Mean Average Precision at 0.5 IoU):** **99.00%**  
* **Precision:** **0.985**  
* **Recall:** **0.980**

This framework significantly outperformed standard YoloV5 and YoloV8 models in both detection accuracy and processing speed, making it highly effective for real-time applications.

![][image4]

# Optimized MoblieNet

https://www.nature.com/articles/s41598-025-27393-z  
**Optimised MobileNet for very lightweight and accurate plant leaf disease detection** 

**V^2PlantNet**

* Using MobileNet CNN architecture  
* Designed for :Real-time agricultural applications, mobile/embedded deployment, low computational cost, high classification accuracy across many crop types 

**Difference compared to original MoblieNet architecture**

*  Batch Normalization (BN) after every convolutions  
* ReLu activation  
* Multi-stage feature extraction  
* Optimized depthwise \+ pointwise convolution blocks   
* Number of parameters: From M\*N\*C\*K to M\*N\*C+1\*\!\*C\*K → 382246 parameters

**Dataset:**

* PlantVillage  
* 70% for training, 15% for validation, 15% for testing

**Architecture**

| Stage | Layer Type | Output Configurations | Repeat |
| :---- | :---- | :---- | :---- |
| Initial | Input | (None, 224, 224, 3\) Input shape | 1 |
| 1 | Conv2D | (None, 112, 112, 32\) 3x3 conv, stride-2, BN, ReLU | 1 |
| 1 | Max Pooling | (None, 56, 56, 32\) 3x3 pool, stride-2 | 1 |
| 1 | Depthwise Conv2D | (None, 56, 56, 32\) 3x3 conv, BN, ReLU | 1 |
| 1 | Conv2D | (None, 56, 56, 64\) 1x1 conv, BN, ReLU | 1 |
| 2 | Depthwise Conv2D | (None, 56, 56, 64\) 3x3 conv, BN, ReLU | 2 |
| 2 | Conv2D | (None, 28, 28, 128\) 1x1 conv, BN, ReLU, stride-2 | 1 |
| 2 | Depthwise Conv2D | (None, 28, 28, 128\) 3x3 conv, BN, ReLU | 5 |
| 3 | Conv2D | (None, 14, 14, 256\) 1x1 conv, BN, ReLU, stride-2 | 1 |
| 3 | Depthwise Conv2D | (None, 14, 14, 256\) 3x3 conv, BN, ReLU | 2 |
| 3 | Conv2D | (None, 14, 14, 256\) 1x1 conv, BN, ReLU | 1 |
| Final | Global Average Pooling | (None, 256\) | 1 |
| Final | Fully Connected | (None, 256), L2, L1 regularization | 1 |
| Final | Dropout | (None, 256), rate=0.45 | 1 |
| Final | Fully Connected | (None, 15), Softmax | 1 |

* The model improves upon MoblieNet V1 by integrating Batch Normalization and ReLu activation after each convolutional layers  
* This stabilizes the training process, reduces covariate shift, and enables faster convergence  
* ReLU further introduces the non-linearity required for learning more complex patterns  
* The model otherwise retain the core structure of MobileNet V1

**Experimental setup and procedure**

* The experiments were conducted on Google Colab, utilizing cloud-based resources, including an Intel Xeon CPU @ 2.20GHz, 12.72 GB of RAM, and an NVIDIA T4 GPU, with 358 GB of disk space available for data storage and outputs.  
* The implementation was carried out in Python (version 3.7) with key libraries such as TensorFlow (2.15.0), Keras, NumPy, Pandas, Matplotlib, Seaborn, and OpenCV.  
* Data retrieval was facilitated via Kaggle’s API, providing secure and efficient access to the datasets.

**Result**

* About 97% to 99% for measuring statistics such as accuracy, recall, F1-score, sensitivity, specificity, MCC  
* Some classes with worse results: Healthy potato leaves, maize diseases, attributed to class imbalance and visual similarity between disease patterns

![][image5]  
**Ablation study:**  
![][image6]  
Batch Normalization layer is the most important part of V^2PlantNet  
**Comparison with other models:**   
![][image7]  
![][image8]  
![][image9]  
V^2PlantNet performed better than SOTA models

# Models

## **1\. Current State-of-the-Art (SOTA) Models**

Recent studies have established new standards in accuracy and multi-class recognition on large-scale datasets:

* **Classification Architectures:** Models such as EfficientNet (B4, B5, V2\_m), DenseNet-77/121, and ResNet-50 are currently leading in terms of accuracy. Specifically, EfficientNet-B4 achieved up to 99.97% accuracy on the PlantVillage dataset, while DenseNet-77 reached an impressive 99.98%. Variants such as Res2Next50 also achieved excellent results (99.85%) thanks to their ability to process multi-scale features.  
* **Object Detection Architectures:** Faster R-CNN (combined with Res2Net or FPN) and Mask R-CNN are considered SOTA approaches for accurately locating diseased regions (lesions) and segmenting individual disease spots. For real-time detection, YOLOv8 and the latest YOLOv12 demonstrate outstanding performance in both speed and accuracy.  
* **Anchor-free Algorithms:** Models such as CenterNet and CornerNet are becoming a new trend because they identify diseases by predicting key points, allowing the models to operate more robustly under leaf occlusion conditions.


## **2\. Lightweight Models and Algorithms**

These models are especially important for deployment on mobile devices, drones, or IoT systems in agricultural fields with limited computational resources:

* **MobileNet Family (V1, V2, V3):** These architectures are standard lightweight models that utilize depthwise separable convolutions to significantly reduce computational cost and parameter size without greatly sacrificing accuracy. The improved MobileNet-Beta version achieved accuracy up to 99.85%.  
* **EfficientNet-B0:** By employing a compound scaling mechanism to balance depth, width, and resolution, EfficientNet-B0 provides highly efficient computation for applications such as the “AgroAId” system.  
* **SqueezeNet and SqueezeNext:** These models are designed with extremely small model sizes, making them suitable for limited hardware such as Raspberry Pi and embedded devices. SqueezeNext further integrates coordinate attention mechanisms to improve feature extraction from diseased leaf regions.  
* **ShuffleNet:** By using channel shuffling techniques to reduce communication costs between layers, ShuffleNet achieves around 96% accuracy in maize disease recognition tasks.  
* **Tiny YOLO Variants:** Algorithms such as Tiny-YOLOv4 and MobileNetV2-YOLOv3 are optimized for fast inference speed, enabling real-time crop disease monitoring with very high mAP values (up to 99.9% on certain datasets).  
* **Model\_Lite:** A simplified variant of ResNet18, optimized to contain only 1/344 of the original model’s parameters while still maintaining accuracy above 91%.

**3\. Important Trends and Supporting Techniques**

Beyond core architectures, combining the following techniques makes models more intelligent and robust:

* **Hybrid Models:** These approaches combine Deep Learning for feature extraction with traditional Machine Learning (ML) algorithms for classification. For example, VGG19 combined with kNN achieved 99.1% accuracy on custard apple leaves, while Inception v3 combined with SVM produced excellent results on banana leaf datasets.  
* **Attention Mechanisms:** Integrating modules such as CBAM, SE, or Coordinate Attention helps models focus on diseased leaf regions while suppressing environmental noise such as soil, weeds, or varying lighting conditions.  
* **Transfer Learning:** Most current SOTA models use pretrained weights from the ImageNet dataset, enabling faster convergence and higher accuracy even when plant disease datasets are limited.  
* **Real-world Challenges:** Practical deployment still faces difficulties related to processing latency (often exceeding 200 ms) and generalization capability when encountering new diseases or complex outdoor lighting conditions.

# Data

**Summary about Data**

# s10462-025-11234-6

## 1\. System Architecture

The plant leaf disease detection system proposed in the research is based on a combination of image processing and machine learning techniques. The architecture is designed to automatically detect diseased regions on plant leaves and classify disease categories accurately.

The system consists of four major stages:

### 1.1 Preprocessing

The input tomato leaf images undergo initial processing to improve quality for subsequent analysis.

* Techniques: Image resizing (to 256 × 256 pixels) and Histogram Equalization (HE).  
* Purpose: To remove noise, improve contrast, and enhance diseased regions to simplify feature extraction.

### 1.2 Segmentation

This stage isolates the infected regions from the healthy parts of the leaf.

* Techniques: K-means Clustering is used for pixel-based segmentation, followed by Contour Tracing to extract boundaries.  
* Purpose: To improve localization accuracy and reduce background interference.

### 1.3 Feature Extraction

The model extracts significant characteristics that represent the disease patterns.

* Techniques:  
  * DWT (Discrete Wavelet Transform): Analyzes frequency components.  
  * PCA (Principal Component Analysis): Reduces data dimensionality while keeping vital info.  
  * GLCM (Gray Level Co-occurrence Matrix): Extracts texture-based statistical features.  
* Purpose: To capture effective disease patterns and improve classification performance.

### 1.4 Classification Networks

Three major classification methods are compared to predict disease categories:

* SVM (Support Vector Machine): A traditional classifier effective for smaller datasets.  
* K-NN (K-Nearest Neighbor): A distance-based simple learning process.  
* CNN (Convolutional Neural Network): The top performer (99.09% accuracy) due to its automatic feature learning capability.  
* CNN Layers: Input Layer → Convolution Layer → ReLU Activation → Pooling Layer → Fully Connected Layer → Output Layer.

## 2\. Datasets Used in the Research

The research primarily utilizes the PlantVillage dataset, a benchmark for plant disease classification.

### 2.1 PlantVillage Tomato Dataset

The dataset specifically focuses on tomato leaf images across six distinct classes:

* Healthy  
* Mosaic Virus  
* Leaf Mold  
* Yellow Curl  
* Spotted Spider Mite  
* Target Spot

Dataset Characteristics:

* Total Images: 600 images.  
* Distribution: 100 images per class.  
* Environment: High-quality images taken in a controlled laboratory environment.

### 2.2 Other Referenced Datasets

The study also references the application of these methods on other plant types, including:

* Wheat, Banana, Bean, Citrus, and Tea leaves.

## 3\. Evaluation Metrics

The performance of the models is measured using standard statistical metrics.

### 3.1 Accuracy

Measures the overall percentage of correctly classified samples.

* SVM: 89%  
* K-NN: 97.3%  
* CNN: 99.09%

### 3.2 Precision

Measures the reliability of positive predictions.

### 3.3 Recall

Measures the completeness of disease detection.

### 3.4 F1-score

The harmonic mean of Precision and Recall, balancing the two.

### 3.5 Confusion Matrix

A visualization tool used to show correct vs. incorrect predictions across all classes.

##  Bonus: Modern AI Evaluation Metrics

In modern deep learning deployment, additional metrics are often used to evaluate efficiency beyond simple accuracy:

* mAP (Mean Average Precision): Essential for object detection and localization.  
* FPS (Frames Per Second): Determines if the model is fast enough for real-time video processing.  
* Latency: The time taken for a single prediction (critical for mobile apps).  
* FLOPs & Parameters: Measure the computational weight and storage size of the model.  
* Grad-CAM: An Explainable AI (XAI) tool used to visualize which part of the leaf the CNN is "looking at" to make a decision.  
* Deployment Compatibility: Checking if the model runs on Edge devices (like smartphones or IoT sensors) vs. Cloud systems.

# 1-s2.0-S2666285X22000218-main

## **1\. System Architectures**

The plant leaf disease detection system proposed in the research is based on a combination of image processing and machine learning techniques. The architecture is designed to automatically detect diseased regions on plant leaves and classify disease categories accurately.

The system consists of several major stages:

### **Preprocessing**

The input tomato leaf images are resized to 256 × 256 pixels. Histogram Equalization (HE) is applied to improve image quality and enhance contrast.

### **Segmentation**

K-means Clustering is used to segment the infected regions from the leaf image. Contour Tracing is then applied to extract leaf boundaries and shape information.

### **Feature Extraction**

The model uses multiple feature extraction techniques:

* DWT (Discrete Wavelet Transform)  
* PCA (Principal Component Analysis)  
* GLCM (Gray Level Co-occurrence Matrix)

These methods extract important texture and statistical features from diseased leaf regions.

### **Classification Networks**

Three classification methods are used:

* SVM (Support Vector Machine)  
* K-NN (K-Nearest Neighbor)  
* CNN (Convolutional Neural Network)

Among them, CNN achieved the best performance with an accuracy of 99.09%.

The CNN architecture includes:

* Input Layer  
* Convolution Layer  
* ReLU Activation Layer  
* Pooling Layer  
* Fully Connected Layer  
* Output Layer

## **2\. Datasets Used in the Research**

The research mainly uses the PlantVillage tomato leaf dataset.

### **PlantVillage Tomato Dataset**

The dataset contains tomato leaf images with six classes:

* Healthy  
* Mosaic Virus  
* Leaf Mold  
* Yellow Curl  
* Spotted Spider Mite  
* Target Spot

Dataset characteristics:

* 600 total images  
* 100 images per class  
* Images resized to 256 × 256 pixels

The dataset is used for both training and testing the classification models.

### **Other Datasets Mentioned in Referenced Studies**

The research also references studies using:

* Wheat Leaf Dataset  
* Banana Leaf Dataset  
* Bean Disease Dataset  
* Citrus Dataset  
* Tea Leaf Dataset

## **3\. Evaluation Metrics**

The research uses several evaluation metrics to measure model performance.

### **Accuracy**

Measures the percentage of correctly classified samples.

Results:

* SVM: 89%  
* K-NN: 97.3%  
* CNN: 99.09%

**Precision**

Measures how many predicted positive samples are actually correct.

Precision=TP/(TP+FP)×100

### **Recall**

Measures how many actual positive samples are correctly detected.

Recall=TP/(TP+FN)×100

### **F1-score**

Represents the harmonic mean of Precision and Recall.

F1=2×Precision×Recall/(Precision+Recall)×100

**Bonus:** 

In many modern AI and deep learning systems, additional evaluation metrics are commonly used even though they are not included in this research.

### **Mean Average Precision (mAP)**

Widely used in object detection tasks to evaluate localization and classification performance simultaneously.

### **Frames Per Second (FPS)**

Measures real-time processing speed of the model.

### **Throughput**

Measures how many samples/images can be processed per second under workload conditions.

### **Latency**

Measures the response time required for one prediction.

### **FLOPs (Floating Point Operations)**

Measures computational complexity of the model.

### **Number of Parameters**

Represents model size and memory requirements.

### **Storage Size**

Measures the disk space required to store the trained model.

### **Power Consumption**

Important for mobile, embedded, and IoT deployment.

### **Grad-CAM / Attention Visualization**

Used in Explainable AI (XAI) to visualize which image regions influence model predictions.

### **Deployment Compatibility**

Evaluates whether the model can run efficiently on:

* mobile devices  
* edge devices  
* cloud systems  
* IoT platforms

# fpls-16-1637241

## **1\. System Architectures**

Deep learning-based plant leaf disease detection systems in the research generally consist of three major architectures:

* two-stage detection networks  
* one-stage detection networks  
* anchor-free detection networks

These architectures are designed to automatically detect diseased regions on plant leaves and classify disease categories accurately.

### **1.1 Two-stage Detection Networks**

The research discusses:

* R-CNN  
* Fast R-CNN  
* Faster R-CNN  
* Mask R-CNN

Characteristics:

* generate candidate regions first  
* perform disease classification afterward  
* high detection accuracy  
* better small-object detection  
* slower processing speed

### **1.2 One-stage Detection Networks**

The research discusses:

* YOLO  
* YOLOv3  
* YOLOv4  
* YOLOv5  
* YOLOv7  
* YOLOv8  
* YOLOv12  
* SSD

Characteristics:

* direct disease detection and classification  
* real-time processing capability  
* faster inference speed  
* suitable for agricultural deployment

### **1.3 Anchor-free Detection Networks**

The research also discusses:

* CenterNet  
* CornerNet  
* YOLOX  
* RT-DETR  
* CoffeeNet

Characteristics:

* no anchor box requirement  
* simplified detection process  
* improved computational efficiency  
* strong robustness in complex environments

### **1.4 Classification Networks**

The paper additionally introduces several CNN-based classification architectures:

* AlexNet  
* VGG16  
* VGG19  
* GoogLeNet  
* ResNet  
* DenseNet  
* EfficientNet  
* MobileNet  
* CapsNet

These models are mainly used for:

* disease classification  
* feature extraction  
* lightweight deployment

## **2\. Datasets Used in Plant Leaf Disease Detection**

The research discusses many publicly available plant disease datasets used for deep learning-based disease detection.

### **2.1 PlantVillage Dataset**

PlantVillage is one of the most widely used datasets.

It contains:

* 54,036 images  
* 14 plant species  
* 26 diseases  
* 38 total categories

However:

* most images are collected under controlled laboratory conditions  
* fewer images are collected in real natural environments

### **2.2 PlantDoc Dataset**

PlantDoc contains:

* 2,598 images  
* 13 plant species  
* 30 classes

The dataset includes images collected from real-world environments.

### **2.3 Other Important Datasets**

The research also discusses:

* Plant Pathology 2020-FGVC7 Dataset  
* Rice Diseases Dataset  
* Corn Leaf Diseases Dataset  
* Durian Leaf Disease Dataset  
* Banana Leaf Spot Dataset  
* Potato Disease Dataset  
* Strawberry Disease Dataset  
* Sugarcane Disease Dataset  
* PDD271 Dataset  
* PDDB Dataset  
* PlantifyDr Dataset  
* Jute Leaf Disease Dataset

These datasets vary in:

* dataset size  
* image quality  
* environmental complexity  
* disease categories  
* annotation quality  
* acquisition methods

Some datasets are collected using:

* cameras  
* drones  
* mobile phones  
* field imaging systems

## **3\. Evaluation Metrics**

Evaluation metrics are essential for measuring the performance of plant disease detection systems.

The research identifies several commonly used metrics:

### **Accuracy**

Measures the percentage of correctly classified samples.

### **Precision**

Measures how many predicted positive samples are actually correct.

### **Recall**

Measures how many actual positive samples are successfully detected.

### **F1-score**

Represents the harmonic mean of Precision and Recall.

### **IoU (Intersection over Union)**

Measures overlap between predicted and actual bounding boxes.

IoU=Area of OverlapArea of UnionIoU=\\frac{Area\\ of\\ Overlap}{Area\\ of\\ Union}IoU=Area of UnionArea of Overlap​

### **Mean Average Precision (mAP)**

Widely used in object detection tasks for evaluating localization and classification performance.

### **FPS (Frames Per Second)**

Measures real-time detection speed.

### **Training Time**

Measures computational training efficiency.

### **Model Size**

Measures storage requirements and lightweight capability.

## **Bonus**

Some evaluation metrics are commonly used in AI systems but are not heavily emphasized in the research.

### **Throughput**

Measures the number of images processed per second.

### **Latency**

Measures prediction response time under workload conditions.

### **FLOPs**

Measures computational complexity of the model.

### **Number of Parameters**

Represents model complexity and memory usage.

### **Power Consumption**

Important for mobile devices and edge AI systems.

### **Grad-CAM Score**

Used for explainable AI and disease region visualization.

### **Attention Visualization**

Shows which image regions influence model predictions.

### **Deployment Compatibility**

Measures whether models can be deployed efficiently on:

* mobile devices  
* embedded systems  
* edge AI devices  
* IoT platforms

# s41598-024-72197-2

# **1\. System Architectures**

The proposed plant leaf disease detection system is based on a hybrid artificial intelligence framework that combines deep learning and machine learning techniques. The architecture is designed to automatically extract discriminative features from plant leaf images and accurately classify various disease categories.

The system consists of several major stages:

## **Data Input**

The system accepts plant leaf images from four different crop datasets, including Banana, Custard Apple, Fig, and Potato leaves. These images contain both healthy and diseased samples collected under real-world conditions.

## **Deep Feature Extraction**

Two pre-trained convolutional neural networks are employed as feature extractors:

### **VGG19**

VGG19 is a deep convolutional neural network consisting of 19 weighted layers. It uses 3 × 3 convolution filters and accepts input images of size 224 × 224 × 3\. VGG19 is highly effective in learning fine-grained texture and structural patterns from leaf images.

### **Inception v3**

Inception v3 is a more advanced architecture that applies factorized convolutions and auxiliary classifiers to reduce computational cost while maintaining high feature extraction capability. It is particularly effective at capturing complex disease patterns.

## **Feature Vector Generation**

The final layers of the pre-trained CNN models are removed, and the deep representations generated by the networks are used as feature vectors. These vectors summarize important visual characteristics such as texture, shape, and color variations associated with plant diseases.

## **Machine Learning Classifiers**

The extracted deep features are classified using five traditional machine learning algorithms:

* Support Vector Machine (SVM)  
* k-Nearest Neighbors (kNN)  
* Random Forest (RF)  
* AdaBoost  
* Decision Tree (DT)

## **Classification Output**

The classifier predicts the disease class of each input image and determines whether the leaf is healthy or affected by a specific disease.

## **Validation Strategy**

Stratified 10-Fold Cross Validation is used to evaluate the models. This technique preserves the class distribution in every fold and provides reliable performance estimates.

## **Best Performing Architectures**

The optimal model differs depending on the dataset:

* Banana Leaf: Inception v3 \+ SVM  
* Custard Apple Leaf and Fruit: VGG19 \+ kNN  
* Fig Leaf: VGG19 \+ kNN  
* Potato Leaf: Inception v3 \+ SVM

# **2\. Datasets Used in the Research**

The study uses four publicly available datasets obtained from [Mendeley Data](https://data.mendeley.com?utm_source=chatgpt.com).

## **Banana Leaf Dataset**

This dataset contains 408 images collected from agricultural fields in Dhaka, Bangladesh.

It includes seven classes:

* Black Sigatoka  
* Bract Mosaic Virus  
* Healthy Leaf  
* Insect Pest  
* Moko  
* Panama  
* Yellow Sigatoka

## **Custard Apple Leaf and Fruit Dataset**

This dataset contains 8,226 images collected in Pune, India.

It includes six classes:

* Anthracnose  
* Blank Canker  
* Diplodia Rot  
* Leaf Spot on Fruit  
* Leaf Spot on Leaf  
* Mealy Bug

## **Fig Leaf Dataset**

This dataset contains 2,321 images collected from different regions of Iraq.

It consists of two classes:

* Healthy  
* Infected

## **Potato Leaf Dataset**

This dataset contains 3,076 images of size 1500 × 1500 pixels collected in Central Java, Indonesia.

It includes seven classes:

* Bacteria  
* Fungi  
* Healthy  
* Nematode  
* Pest  
* Phytophthora  
* Virus

## **Dataset Summary**

* Banana Leaf Dataset: 408 images, 7 classes  
* Custard Apple Dataset: 8,226 images, 6 classes  
* Fig Leaf Dataset: 2,321 images, 2 classes  
* Potato Leaf Dataset: 3,076 images, 7 classes

Total number of images: 14,031

# **3\. Evaluation Metrics**

The study employs several evaluation metrics to assess the performance of the proposed models.

## **Accuracy**

Measures the proportion of correctly classified samples among all samples.

Accuracy=TP+TN/(TP+TN+FP+FN)

**Precision**

Measures how many predicted positive samples are actually correct.

Precision=TP/(TP+FP)×100

## **Recall**

Measures how many actual positive samples are correctly identified.

Recall=TP/(TP+FN)×100

## **F1-Score**

Represents the harmonic mean of Precision and Recall.

F1=2×Precision×Recall/(Precision+Recall)×100

## **Area Under the Curve (AUC)**

Measures the area under the ROC curve and evaluates the model’s ability to distinguish between classes across different thresholds.

## **Matthews Correlation Coefficient (MCC)**

Provides a robust single-value metric that remains reliable even when the dataset is imbalanced.

![][image10]

## **Confusion Matrix**

Summarizes classification outcomes using:

* TP (True Positive)  
* TN (True Negative)  
* FP (False Positive)  
* FN (False Negative)

# **Bonus: Additional Evaluation Metrics Commonly Used in AI**

In modern artificial intelligence and deep learning systems, several additional evaluation metrics are frequently used even though they are not included in this research.

## **Balanced Accuracy**

Calculates the average recall across all classes and is particularly useful for imbalanced datasets.

## **Cohen’s Kappa**

Measures agreement between predicted and true labels while accounting for chance agreement.

## **Log Loss (Cross-Entropy Loss)**

Evaluates the confidence of probabilistic predictions.

## **Top-k Accuracy**

Determines whether the correct label appears among the top k predictions.

## **Mean Average Precision (mAP)**

Widely used in object detection to jointly evaluate classification and localization performance.

## **Intersection over Union (IoU)**

Measures overlap between predicted and ground-truth regions in detection and segmentation tasks.

## **Dice Coefficient**

A common metric for image segmentation that quantifies region similarity.

## **FLOPs (Floating Point Operations)**

Estimates the computational complexity of a model.

## **Number of Parameters**

Indicates model size and memory requirements.

## **Latency**

Measures the time required to process a single input.

## **Throughput**

Measures how many samples can be processed per second.

## **Frames Per Second (FPS)**

Evaluates real-time inference speed, especially in video applications.

## **Power Consumption**

Important for deployment on mobile, embedded, and IoT devices.

## **Grad-CAM and Attention Visualization**

Used in Explainable AI (XAI) to highlight image regions that most influence model predictions.

# Tổng hợp

## **1\. System Architectures**

The research identifies three distinct architectural approaches to plant disease detection, ranging from traditional hybrid models to advanced deep learning frameworks.

### **1.1 Hybrid Machine Learning Architecture**

This architecture combines deep feature extraction with classical statistical classifiers.

* **Feature Extraction:** Utilizes pre-trained CNNs like **VGG19** or **Inception v3** (with final layers removed) to generate deep representation vectors summarizing texture, shape, and color.  
* **Classification:** Employs algorithms such as **SVM, k-NN, Random Forest, AdaBoost,** and **Decision Trees**.  
* **Preprocessing:** Includes **Histogram Equalization (HE)** for contrast enhancement and **K-means Clustering** for isolating infected regions (Segmentation).

### **1.2 Deep Learning Detection Networks**

These architectures are designed for simultaneous localization (bounding boxes) and classification.

* **Two-stage Detectors:** Models like **Faster R-CNN** and **Mask R-CNN** that generate region proposals before classification, offering high accuracy for small-scale disease spots.  
* **One-stage Detectors:** Real-time models including the **YOLO series (v5, v7, v8, up to v12)** and **SSD**, optimized for speed and agricultural field deployment.  
* **Anchor-free Detectors:** Systems like **CenterNet** or **RT-DETR** that simplify the detection process by removing the need for predefined anchor boxes.

### **1.3 End-to-End Classification Networks**

Deep CNN architectures focused on high-accuracy category prediction.

* **Standard Models:** **AlexNet, VGG16/19, GoogLeNet, ResNet,** and **DenseNet**.  
* **Lightweight Models:** **MobileNet** and **EfficientNet**, designed for low-power mobile or edge AI environments.  
* **Architecture Flow:** Input Layer → Convolutional Layers (Feature Learning) → ReLU/Activation → Pooling → Fully Connected Layers → Softmax Output.

## **2\. Datasets Used in the Research**

The papers reference a wide variety of datasets, ranging from controlled laboratory environments to diverse real-world field conditions.

### **2.1 Benchmark Datasets**

* **PlantVillage:** The most widely cited dataset (54,305+ images, 14 species, 38 classes). While high-quality, it is noted for being collected in controlled settings.  
* **PlantDoc:** A smaller but more realistic dataset (approx. 2,598 images) featuring leaves in natural environments with complex backgrounds.

### **2.2 Crop-Specific Datasets**

The research covers a broad spectrum of agricultural products:

* **Vegetables & Fruits:** Tomato (6-10 classes including Mosaic Virus, Leaf Mold, Yellow Curl), Potato (Bacteria, Fungi, Phytophthora), Fig, and Custard Apple.  
* **Field & Plantation Crops:** Wheat, Banana (Black Sigatoka, Moko, Panama disease), Rice, Corn, Citrus, Tea, and Sugarcane.  
* **Specialty Crops:** Coffee, Strawberry, Durian, and Jute.

### **2.3 Dataset Characteristics**

* **Acquisition Methods:** Images captured via digital cameras, mobile phones, drones, and laboratory imaging systems.  
* **Annotation:** Includes image-level labels for classification and bounding-box/pixel-level masks for detection and segmentation tasks.

## **3\. Evaluation Metrics**

Performance is measured through a combination of statistical accuracy and operational efficiency.

### **3.1 Classification Performance Metrics**

* **Accuracy:** Overall percentage of correct predictions. (Top models like CNN often exceed 99%).  
* **Precision & Recall:** Precision measures the reliability of positive hits; Recall measures the ability to find all diseased instances.  
* **F1-Score:** The harmonic mean of Precision and Recall, used to balance the two in imbalanced datasets.  
* **MCC (Matthews Correlation Coefficient):** Provides a more robust evaluation than F1-score for binary and multiclass classification.

### **3.2 Detection & Localization Metrics**

* **IoU (Intersection over Union):** Measures the overlap between the predicted bounding box and the ground truth.  
* **mAP (Mean Average Precision):** The standard metric for object detection, calculating the average precision across all disease categories.

### **3.3 Operational & Bonus Metrics**

For real-world agricultural deployment (Mobile/IoT), the following are essential:

* **Inference Speed:** Measured in **FPS (Frames Per Second)** or **Latency** (ms per image).  
* **Model Complexity:** Evaluated by the **Number of Parameters** and **FLOPs** (Floating Point Operations).  
* **Explainable AI (XAI):** Using **Grad-CAM** or **Attention Maps** to visualize the specific leaf regions influencing the model’s decision.  
* **Deployment Compatibility:** Assessment of performance on edge devices versus cloud-based systems.

# Outline

# Resources

**Papers:**  
[Advancing\_Plant\_Detection\_ML\_DL(CNN).pdf](https://drive.google.com/file/d/1Lt34GL9LpCYAYEe-ABkEO6UtD08JlKP7/view?usp=sharing)  
[CNN\_Detection\_Classification\_Review.pdf](https://drive.google.com/file/d/152sDGm0Iy6kQTSTuqxOgdg8a-yb4ljvQ/view?usp=sharing)  
[CV\_ML\_Detection(KNN,CNN, SVM).pdf](https://drive.google.com/file/d/1Z1njHcbcCJ32Vh8Qh7J5ttEG7f6jgQ2r/view?usp=sharing)  
[Review\_On\_DL\_In\_PDD.pdf](https://drive.google.com/file/d/1y__xrQHmS0Re3u7xJl05xBF36Zolyu-y/view?usp=sharing)

[IJCESEN](https://drive.google.com/file/d/19lCnQU9087FhusGMhS8QSxcyiG1WyT5m/view?usp=drive_link)  
[DenseNet201Plus: Cost-effective transfer-learning architecture for rapid leaf disease identification with attention mechanisms](https://drive.google.com/file/d/118Q9h8cOKovURHKIEXntmXlYqAm8l3t3/view?usp=sharing)  
[s10343-022-00796-y.pdf](https://drive.google.com/file/d/1LFNIodqIU7ZEN5y3shVfvQH7mx_pDnoS/view?usp=sharing)

# Tasks

- W3: Tổng quan dự án

# Week1

# Dataset

- Ảnh data bị mờ \-\> dùng abc giải quyết  
- 

# Model

- Model nó bị quá nhỏ để data \-\> Scale

- src  
  - Resnet  
  - ViT  
- utlis  
  - load\_data  
  - model\_helper  
  - args  
- main.py

# Week3

# Models

- Survey các kiến trúc liên quan đến định hướng của mình trong các bài survey, chọn ra 2-3 bài có kết quả cao nhất và tổng hợp ra báo cáo trước khi họp tuần sau ( Khôi 1 bài, Kiên 1 bài)

# Datasets

- Xem trước Data: PlantDoc và IDADP. Xem data này gặp những vấn đề gì và người ta sẽ thường xử lý như thế nào, tổng hợp ra báo cáo trước khi họp tuần sau ( Lân với Luân mỗi người 1 bộ)   
- 


  
Họp: T4 tuần sau (20/5)

Họp để:

- Xem báo cáo về Models và Data để cả nhóm hiểu PRJ  
- Cấu trúc Project  
- Từng người sẽ implement gì để bắt đầu code từ tuần sau

# Week4

# **Models**

- Kiên Implement Deep Dive: Advanced Lightweight CNN  
- Khôi: Optimised MobileNet for very lightweight and accurate plant leaf disease detection 

# **Datasets**

- Chuẩn hóa Quy trình xử lý data và code

# Checklist

Các em cần chuẩn bị Google driver của AI project, cập nhật dần cho đến Deadline nộp quyền để đánh giá điểm bao gồm:

- [ ] Báo cáo  word file/ latex files và pdf file theo mẫu \- Long  
- [ ] Vắn tắt hướng dẫn cài đặt, môi trường cài đặt \- Kiên   
- [x] ~~Ppt trình bày đồ án~~  
- [ ] Video demo các case study, mỗi video / case study; VD:   
      - [ ] Video training model \- Long  
      - [ ] video trình bày case studies cho mỗi use case(Demo). \- Lân, Luân  
      - [ ] Video hướng dẫn cài đặt, thiết lập môi trường \- Kiên   
- [ ] Codes, dữ liệu, Database (có hướng dẫn readme.txt cài đặt, các gói phần mềm..etc) \- Long   
- [x] ~~Dữ liệu thực nghiệm, các dữ liệu khác \- Khôi~~  
- [x] ~~Các bài báo/ tạp chí  viết cho project, bài báo  tham khảo vào một folder \- Khôi~~

Mỗi mục trên là 1 thư mục gửi link trong google

Chi tiết:  
Hdan cài đặt môi trường có thể nhờ AI Viết, kiểm tra lại rồi đẩy lên drive  
Lân với Luân sửa xong app thì quay video demo luôn  
Kiên làm xong hdan cài đặt môi trường xong tự quay luôn  
Khôi ngồi tải dữ liệu của PlantVillage với PlantDoc xuống rồi up lên drive vào thư mục, link tải dữ liệu t để trên Repo  
Tải luôn các bài báo mình sử dụng ( T để link bài báo trong PPT trình bày rồi, bấm vào link rồi tải về up lên thôi)

# Architecture & Model Implementations

This project benchmarks lightweight Deep Learning architectures for both plant leaf disease classification and object detection, evaluating their predictive capabilities and resource footprints.

# CNN Lightweight

![][image11]  
The proposed lightweight CNN architecture is built upon the MobileNetV2 backbone to achieve an effective trade-off between computational efficiency and feature representation for multi-class plant disease classification. The network first extracts hierarchical visual features using the MobileNetV2 backbone, followed by additional depth-wise separable convolution layers that factorize standard convolutions into depth-wise and point-wise operations. Furthermore, grouped convolutions with a group size of G \= 4 are employed to further reduce FLOPs and the number of trainable parameters while maintaining sufficient channel interaction.

To enhance feature discrimination, an enhanced Squeeze-and-Excitation (SE) block is integrated after the depth-wise separable convolution stage. Unlike conventional SE modules, the proposed design incorporates Group Normalization (GN), which provides more stable feature normalization under small-batch training conditions and improves channel-wise feature recalibration. The architecture also includes modified residual skip connections that utilize depth-wise convolution operations within the shortcut path, enabling efficient gradient propagation while preserving the lightweight nature of the model.

Finally, the extracted feature maps are passed through fully connected layers followed by a dropout layer to improve generalization and reduce overfitting. A Softmax output layer then produces probability distributions over 38 plant disease categories, allowing the model to perform multi-class classification of plant species and health conditions.

# YOLO LeafNet

![][image12]

The network architecture of YOLO-LeafNet is structured as a single-stage object detection model designed to execute feature extraction, bounding box regression, and multi-class probability estimation within a single forward pass. The physical topology of the framework comprises three sequential structural components: a backbone network, a neck network, and a multi-scale prediction head. 

The backbone of the proposed YOLO-LeafNet is derived from the YOLOv8 architecture, utilizing CSPDarknet53 as the primary feature extraction network. The backbone consists of multiple convolutional (Conv) layers and C2f modules that progressively extract hierarchical features from the input leaf image. To further improve feature representation and model generalization, two additional components are incorporated into the backbone: Batch Normalization and Dropout layers. Batch Normalization stabilizes the learning process by reducing internal covariate shifts and accelerating convergence, while the Dropout layer mitigates overfitting by randomly deactivating neurons during training. In addition, a Spatial Pyramid Pooling (SPP) module is employed at the end of the backbone to capture contextual information at multiple receptive fields, enabling the network to effectively recognize leaf disease patterns of varying sizes and scales. 

The neck of YOLO-LeafNet adopts a Feature Pyramid Network (FPN) structure to facilitate multi-scale feature fusion. This component receives feature maps from different stages of the backbone and combines them through a sequence of upsampling and concatenation operations. The fused features are subsequently refined using C2f blocks, which enhance information propagation and preserve important semantic details across scales. By integrating both high-level semantic information and low-level spatial details, the neck improves the model’s ability to detect disease symptoms appearing at different resolutions within leaf images. 

The detection head follows an anchor-free design inherited from YOLOv8, allowing the model to directly predict object locations and class probabilities without relying on predefined anchor boxes. The head performs multi-scale detection using three detection branches corresponding to small, medium, and large feature maps. Specifically, detection is conducted at resolutions of 72×72, 36×36, and 18×18, enabling the network to identify disease regions of different sizes. Each detection branch consists of convolutional layers responsible for predicting bounding box coordinates, object confidence scores, and disease class probabilities. This multi-scale detection strategy enhances localization accuracy and improves recognition performance for diverse leaf disease manifestations. 

# V2PlantNet

V2PlantNet is a lightweight convolutional neural network designed for plant disease classification with a strong focus on computational efficiency. The architecture follows a MobileNet-inspired design, replacing conventional convolutions with **depthwise separable convolutions** to significantly reduce the number of trainable parameters and computational operations while preserving effective feature extraction.

The network begins with an **initial 3x3 convolutional layer** followed by **Batch Normalization (BN)** and a **ReLU activation function**. This stage extracts low-level visual features such as edges, textures, and color patterns from the input image. A **Max Pooling layer** then reduces the spatial resolution of the feature maps, decreasing the computational cost for subsequent layers.

The main feature extraction process consists of three stages of **Depthwise Separable Blocks** with increasing channel dimensions. Each block separates a standard convolution into two operations:

* **Depthwise Convolution:** Applies an independent 3×3 convolution filter to each input channel to capture spatial information.  
* **Pointwise Convolution:** Applies a 1×1 convolution to combine information across channels and generate new feature representations.

By separating spatial feature extraction and channel mixing, depthwise separable convolutions greatly reduce the number of parameters and floating-point operations (FLOPs) compared to standard convolutions.

The first stage contains three depthwise separable blocks with **64 output channels**, capturing more detailed local features while maintaining the spatial resolution. The second stage consists of seven blocks with **128 output channels**, where the first block performs downsampling to reduce the feature map size and increase the receptive field. The third stage contains three blocks with **256 output channels**, allowing the network to learn more abstract and high-level disease characteristics.

After feature extraction, a **Global Average Pooling (GAP)** layer is applied to transform the final feature maps into a compact feature vector. GAP reduces the number of parameters compared to traditional fully connected layers and helps reduce overfitting by summarizing the spatial information of each feature channel.

The resulting feature vector is passed through a **fully connected layer with 256 neurons**, followed by a **Dropout layer with a dropout rate of 0.45** to improve generalization by randomly disabling a fraction of neurons during training. Finally, the output classification layer produces prediction scores (logits) corresponding to the **38** plant disease categories. During inference, a Softmax function converts these logits into class probabilities, and the class with the highest probability is selected as the final prediction.

Overall, V2PlantNet achieves a balance between accuracy and efficiency by combining depthwise separable convolutions, progressive channel expansion, global average pooling, and dropout regularization. The model requires approximately **0.379 million parameters** and **0.183 GFLOPs** in the current implementation, making it suitable for deployment on devices with limited computational resources.

# Application Demonstration

Beyond benchmarking, the selected detector was integrated into a desktop diagnostic application implemented with CustomTkinter. The application provides a user-friendly interface designed for real-world crop monitoring.

Key features of the application include:
* **Image Selection and Visualization:** Support for JPEG, PNG, and WebP files, with side-by-side display of the original uploaded image and the annotated YOLO detection result.
* **Local YOLOv12s Inference:** Runs the lightweight YOLOv12s object detector locally to highlight disease lesions with bounding boxes, predicted class names, and confidence scores.
* **Explicit No-Symptom Feedback:** Automatically displays structured status notifications when no disease symptoms are detected.
* **Dynamic Bilingual Support (English & Vietnamese):** Toggled via a segmented button control, instantly updating all user interface labels, status text, and textbox placeholders.
* **Direct Groq API Integration (Llama 3.1):** Leverages `llama-3.1-8b-instant` through the Groq library. The model uses an embedded pre-configured key for zero-setup execution, eliminating manual environment setup.
* **Dynamic Prompt Localization:** The advisory prompt is constructed dynamically based on the active language setting. This directs the model to output plant disease diagnoses, probable causes, and prevention guidelines in either English or Vietnamese.
* **Background Thread Execution:** Model inference and remote API consultations are handled asynchronously on background threads to ensure the UI remains responsive and does not freeze during remote requests.

