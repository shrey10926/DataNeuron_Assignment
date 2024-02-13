# Semantic Similarity API

## Overview
This repository contains a Flask API for calculating semantic similarity between pairs of sentences using a Deep Learning approach.

## How It Works
The API utilizes a fine-tuned sentence-transformer model. Transformer models, such as BERT, RoBERTa, or DistilBERT, are known for their effectiveness in various NLP tasks, including semantic similarity calculations. The model takes two input sentences and outputs a similarity score, indicating the degree of similarity between them.

## Tech Stack
Python

Flask

PyTorch

Sentence-Transformer

HuggingFace

Docker

## Approach
1. Part-A

    The dataset provided (.csv file) was first analysed to observe if any cleaning/preprocessing was needed. After analysing the data some preprocessing steps were performed to clean the data.

    Then this cleaned dataset was used to fine tune a sentence transformer model in order to generate embeddings on this custom dataset. The fine tuning code is available in the training.py file.

    This fine-tuned model was then used in the inferencing phase to generate embeddings which were then used to calculate the cosine similarity between the pair of sentences provided in the API request.


2. Part-B

    Flask framework was used in order to develop this API. The same preprocessing steps were used in the inferencing phase as were used during the training phase. api.py contains the API code. After the API was developed, a Docker image of the API was deployed on Azure instance.

## Request body
{"text1" : "nuclear body seeks new tech....","text2" : "terror suspect face arrest...."}

## Response body
{"similarity_score" : 0.2}

## NOTE
Please test the API using the below URL in Postman. Simply open Postman, select the Body option, then select raw and finally select JSON option from the dropdown menu. Then paste the below URL in the URL section and don't forget to select POST. 
The request and response have beed kept as mentioned in the instruction PDF


LIVE API ENDPOINT : http://172.179.1.93:5801/match_text




![Screenshot from 2024-02-13 23-22-45](https://github.com/shrey10926/DataNeuron_Assignment/assets/79737465/3987c9a5-426a-4006-af2c-9c8f30bef01a)
