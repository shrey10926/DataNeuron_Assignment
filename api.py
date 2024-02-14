from sentence_transformers import SentenceTransformer, models, util
import pickle
import pandas as pd
import numpy as np
import os
import re
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)

#define the route which uses the POST method
@app.route('/match_text', methods=['POST'])
def similarity():

    try:

        #get the request data
        data = request.get_json()

        #verify whether request data contains a pair of text
        if 'text1' not in data or 'text2' not in data:
            return jsonify({'error': 'Both text1 and text2 must be provided.'}), 400

        #remove the extra whitespaces
        new_data = {}
        for k, v in data.items():
            v = re.sub(r'\s+', ' ', v)
            new_data[k] = v

        #fetch the sentences
        sentences1 = new_data['text1']
        sentences2 = new_data['text2']

        #vectorize both the pair of sentences
        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)

        #calculate the cosine similarity
        cosine_scores = util.cos_sim(embeddings1, embeddings2)

        return jsonify({'similarity_score': cosine_scores[0][0].item()}), 200

    except Exception as e:
        return jsonify({'error' : str(e)}), 500




if __name__ == '__main__':

    print(f'loading model...')
    word_embedding_model = models.Transformer("sshreyy/final_similarity_model", max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # word_embedding_model = models.Transformer("output/sentence-transformers_all-mpnet-base-v2-2024-02-13_17-18-11", max_seq_length=512)
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    app.run(debug=False, port = 5801, host = '0.0.0.0', threaded = False)