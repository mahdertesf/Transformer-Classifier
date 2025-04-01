# Amharic Transformer: Pre-training and Hate Speech Recognition

This repository contains the code and resources for pre-training a Transformer model from scratch on an Amharic dataset and fine-tuning it for Amharic hate speech recognition.

The goal is to build Transformer models from scratch, pre-train them using Masked Language Modeling (MLM), and fine-tune them on labeled data to develop a model capable of understanding the nuances of the Amharic language and accurately identifying hate speech content.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Pre-training](#pre-training)
- [Fine-tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Introduction

This project explores the application of Transformer networks to the Amharic language, focusing on pre-training and hate speech recognition. Amharic, being a low-resource language, benefits significantly from pre-training techniques that allow models to learn contextualized word representations from large amounts of unlabeled data.

This project leverages a Masked Language Model (MLM) approach for pre-training, followed by fine-tuning on a labeled dataset for hate speech detection. The trained model is then deployed on the **Mahder AI** web application.

### Key Steps:
- Data collection and preprocessing
- Transformer model implementation
- Pre-training using MLM
- Fine-tuning for hate speech recognition
- Evaluation and performance metrics
- Deployment on **Mahder AI**

## Project Structure

```
.
├── data/
│   ├── totaldata.json                  # Unlabeled Amharic text for pre-training
│   ├── original_hate_speech_data/
│   │   ├── labels.txt                   # Hate speech labels (Hate/Free)
│   │   ├── posts.txt                    # Hate speech posts
│   │   ├── labels_binary.txt            # Hate speech labels (0/1)
│   ├── cleaned_data.txt                 # Cleaned pre-training data
│   └── sample.json                      # Sample data
│
├── model_weights/
│   ├── pretrained_model_checkpoint/
│   │   └── model_weights_v3.h5          # Checkpoint of the pre-trained Transformer
│   ├── finetuned_model_weight/
│   │   └── binary_classifier_weights_v21.h5  # Weights after fine-tuning
│   └── model_weights.h5                 # Saved checkpoints
│
├── sentencepiece_model/
│   ├── amharic_sp_model.model           # SentencePiece model file
│   ├── amharic_sp_model.vocab           # SentencePiece vocabulary file
│
└── train.ipynb                # Jupyter notebook with code and documentation
```

## Dataset

### Pre-training Data
A large corpus of unlabeled Amharic text was collected from various Telegram channels (e.g., **Tikvah Ethiopia, Addis Standard Amharic**). The dataset consists of news articles, discussions, and other textual content.
- **Location:** `/data/totaldata.json`

### Fine-tuning Data
A labeled dataset for Amharic hate speech recognition was obtained from Mendeley Data:
- **[Download Dataset](https://data.mendeley.com/datasets/ymtmxx385m/1)**
- **Location:** `/data/original_hate_speech_data/`

#### Data Preprocessing Steps
- Removing URLs, hashtags, mentions, emojis, and English words
- Subword tokenization using **SentencePiece**
- Padding sequences for uniform length
- Creating input and target pairs for MLM pre-training

## Model Architecture

The model is a standard Transformer network with the following key components:

- **Embedding Layer**: Maps tokens to dense vectors.
- **Positional Encoding**: Adds information about token positions in a sequence.
- **Encoder**: A stack of `EncoderLayers`, each consisting of:
  - Multi-head self-attention mechanism
  - Feed-forward network
- **Decoder**: A stack of `DecoderLayers`, each consisting of:
  - Masked multi-head self-attention mechanism
  - Multi-head attention over encoder output
  - Feed-forward network
- **Multi-Head Attention**: Allows the model to attend to different input segments.
- **Feed-Forward Network**: A fully connected network with ReLU activation.
- **Final Dense Layer**: Outputs probabilities over the vocabulary (for MLM) or a binary classification output (for fine-tuning).

### Model Parameters
- `NUM_LAYERS = 6`
- `EMBEDDING_DIM = 512`
- `FULLY_CONNECTED_DIM = 2048`
- `NUM_HEADS = 8`

## Pre-training

The Transformer model is pre-trained on the unlabeled Amharic text data using the **Masked Language Model (MLM)** objective. This involves:

- Randomly masking ~15% of words in each sentence.
- Training the model to predict the masked words based on the surrounding context.
- Utilizing **SentencePiece Tokenizer** to handle Amharic's complex morphology and OOV words.

## Fine-tuning

The fine-tuned model is trained for **binary classification** (Hate/Free) with the following steps:

- Loading the **pre-trained model weights**.
- Removing the decoder and adding a **classification layer** on top of the encoder.
- Training the classification layer (optionally some encoder layers) on the labeled dataset.
- Using a **learning rate of 5e-7** and **BinaryCrossentropy loss** with the Adam optimizer.

## Evaluation

The model is evaluated using a held-out validation set with key metrics:

- **Validation Loss**
- **Validation Accuracy**
- **Training Loss**
- **Training Accuracy**

> **Achieved Validation Accuracy:** **99%** on the fine-tuned dataset.

## Deployment

The fine-tuned model is deployed on the **Mahder AI** web application for real-time Amharic hate speech detection.

- Users input a Telegram channel username.
- The backend collects all the data from that channel, groups the texts into chunks, and feeds them into the model.
- The model returns a classification indicating whether the text contains hate speech and the level of hate or non-hate speech.

For more details about the deployment, see my app repository:
- **[Mahder AI Repo](https://github.com/mahdertesf/Mahder-AI)**
- **[Deployment Code](https://github.com/mahdertesf/Mahder-AI/blob/main/backend/telegramhate/views.py)**

## Conclusion

This project presents a comprehensive approach to building an **Amharic hate speech detection** model using **Transformer networks**. The **pre-training and fine-tuning techniques** provide a viable solution for addressing the challenges of **low-resource language processing**, contributing to the development of responsible AI technologies for the Amharic-speaking community.

## Future Work

- **Expand Pre-training**: Increase the dataset size and training duration on high-performance GPUs.
- **Enhance Fine-tuning Data**: Curate a larger, well-balanced Amharic hate speech dataset.
- **Optimize Model Architectures**: Experiment with different Transformer architectures and hyperparameters.
- **Multilingual Expansion**: Extend support to other Ethiopian languages using multilingual pre-training.
- **Explainability**: Develop explainable AI techniques to highlight words influencing hate speech classification.

---
🚀 **Contributions are welcome!** If you'd like to contribute, feel free to submit a pull request or open an issue.
