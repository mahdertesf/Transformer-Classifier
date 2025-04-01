# Amharic Transformer: Pre-training and Hate Speech Recognition

This repository contains the code and resources for pre-training a Transformer model from scratch on an Amharic dataset and fine-tuning it for Amharic hate speech recognition.  The goal is to develop a model capable of understanding the nuances of the Amharic language and accurately identifying hate speech content.

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

This project explores the application of Transformer networks to the Amharic language, focusing on pre-training and hate speech recognition. Amharic, being a low-resource language, benefits significantly from pre-training techniques that allow models to learn contextualized word representations from large amounts of unlabeled data. This project leverages a Masked Language Model (MLM) approach for pre-training, followed by fine-tuning on a labeled dataset for hate speech detection. The trained model is then deployed on the Mahder AI web app.

This repository includes the following major steps: data collection and preprocessing, Transformer model implementation, pre-training using MLM, fine-tuning for hate speech recognition, evaluation, and deployment on Mahder AI.

## Project Structure
Use code with caution.
Md
.
├── data/
│ ├── totaldata.json # Unlabeled Amharic text for pre-training
│ ├── original_hate_speech_data/
│ │ ├── labels.txt # Hate speech labels (Hate/Free)
│ │ ├── posts.txt # Hate speech posts
│ │ └── labels_binary.txt # Hate speech labels (0/1)
│ ├── cleaned_data.txt # Cleaned pre-training data
│ └── sample.json # Sample data
├── model_weights/
│ ├── pretrained_model_checkpoint/
│ │ └── model_weights_v3.h5 # Checkpoint of the pre-trained Transformer
│ ├── finetuned_model_weight/
│ │ └── binary_classifier_weights_v21.h5 # Weights after Fine-tuning
│ └── model_weights.h5 # Saved Checkpoints
├── sentencepiece_model/
│ ├── amharic_sp_model.model # SentencePiece model file
│ └── amharic_sp_model.vocab # SentencePiece vocabulary file
└── [notebook_name].ipynb # Jupyter notebook with code and documentation

## Dataset

*   **Pre-training Data:** A large corpus of unlabeled Amharic text collected from various Telegram channels (Tikvah Ethiopia, Addis Standard Amharic, etc.).  The dataset consists of news articles, discussions, and other content. See `/data/totaldata.json`
*   **Fine-tuning Data:** A labeled dataset for Amharic hate speech recognition obtained from Mendeley Data: [https://data.mendeley.com/datasets/ymtmxx385m/1](https://data.mendeley.com/datasets/ymtmxx385m/1). See `/data/original_hate_speech_data/`.

The data preprocessing steps include:

*   Cleaning the text by removing URLs, hashtags, mentions, emojis, and English words.
*   Subword tokenization using SentencePiece.
*   Padding sequences for uniform length.
*   Creating input and target pairs for MLM pre-training.

## Model Architecture

The model is a standard Transformer network with the following key components:

*   **Embedding Layer:** Maps tokens to dense vectors.
*   **Positional Encoding:** Adds information about the position of tokens in the sequence.
*   **Encoder:**  A stack of EncoderLayers, each consisting of a multi-head self-attention mechanism and a feed-forward network.
*   **Decoder:** A stack of DecoderLayers, each consisting of a masked multi-head self-attention mechanism, a multi-head attention mechanism over the encoder output, and a feed-forward network.
*   **Multi-Head Attention:** Allows the model to attend to different parts of the input sequence.
*   **Feed-Forward Network:**  A fully connected network with ReLU activation.
*   **Final Dense Layer:**  Maps the decoder output to a probability distribution over the vocabulary (for pre-training) or a single sigmoid output (for fine-tuning).

**Model Parameters:**

*   `NUM_LAYERS = 6`
*   `EMBEDDING_DIM = 512`
*   `FULLY_CONNECTED_DIM = 2048`
*   `NUM_HEADS = 8`

## Pre-training

The Transformer model is pre-trained on the unlabeled Amharic text data using the Masked Language Model (MLM) objective. This involves:

*   Randomly masking a percentage of words in each sentence (approximately 15%).
*   Training the model to predict the masked words based on the surrounding context.
*  **SentencePiece Tokenizer**: The model uses SentencePiece subword tokenization to effectively handle Amharic's complex word structures and out-of-vocabulary words.

## Fine-tuning

After pre-training, the model is fine-tuned on the labeled hate speech dataset for binary classification (Hate/Free). The steps involved are:

*   Loading the pre-trained model weights.
*   Removing the decoder and adding a new classification layer on top of the encoder.
*   Training the classification layer (and optionally, some of the encoder layers) on the labeled dataset.
*   Utilizing a learning rate of 5e-7 and the BinaryCrossentropy loss function with the Adam optimizer.

## Evaluation

The fine-tuned model is evaluated on a held-out validation set to assess its performance.  Key metrics include:

*   Validation Loss
*   Validation Accuracy
*   Training Loss
*   Training Accuracy

*Achieved a validation accuracy of approximately 99% with the finetuned data*

## Deployment

The fine-tuned model is deployed on the Mahder AI web application for real-time Amharic hate speech detection.  Mahder AI allows users to submit Amharic text and receive a prediction indicating whether the text is likely to contain hate speech. (replace this with the correct website)

## Conclusion

This project provides a comprehensive approach to building an Amharic hate speech detection model using Transformer networks.  The pre-training and fine-tuning techniques demonstrate a practical solution for addressing the challenges of low-resource language processing and contribute to the development of responsible AI technologies for the Amharic-speaking community.

## Future Work

*   **Scale Pre-training:**  Increase the scale of the pre-training dataset and train for more epochs using more powerful GPUs to further improve the model's language understanding capabilities.
*   **Improve Fine-tuning Dataset:**  Curate a larger and more diverse Amharic hate speech dataset with careful attention to labeling quality and balance.
*   **Explore Model Architectures:** Investigate different Transformer architectures and hyperparameter settings to optimize the model for hate speech detection.
*   **Multilingual Capabilities:**  Expand the model to handle other Ethiopian languages or incorporate multilingual pre-training techniques.
*   **Explainable AI:** Develop methods to explain the model's predictions and identify the specific words or phrases that contribute to a hate speech classification.
