# IMPORTS
print("Importing")
import warnings
warnings.filterwarnings("ignore")

import argparse
import glob
import os
import os.path as osp
import json

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import joblib

from rouge_score import rouge_scorer
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# train the word2vec model with our cleaned data

# GLOBAL VARIABLES 
import gensim.downloader as api

# word2vec_model = api.load('word2vec-google-news-300')
model_path = "../../GoogleNews-vectors-negative300-SLIM.bin.gz"
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("vars")

# word2vec_model = KeyedVectors.load("pretrained_model.model")
# word2vec_model.save("pretrained_model.model")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

class WCL:
    def __init__(self):
        print("making 1")
        self.lexicon_data = pd.read_csv('../../WCL_data/lexicon.csv')
        self.data = self.lexicon_data.dropna(subset=["word", "rating"]) 
        self.X = self.data["word"].values  # Target words
        self.y = self.data["rating"].values.astype(float)
        self.vectorizer = TfidfVectorizer()
        self.X_vectors = self.vectorizer.fit_transform(self.X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_vectors, self.y, test_size=0.2, random_state=42)

        # Train regressor
        self.regressor = joblib.load("../../WCL_regressor.pkl")


    def simplify_sentence(self, sentence, difficulty_threshold=3):
        """
        Simplifies a sentence by replacing difficult words with simpler alternatives.
        
        Args:
            sentence (str): Input sentence to be simplified.
            regressor: Trained regressor model for predicting word difficulty.
            vectorizer: Trained vectorizer for transforming words into features.
            word2vec_model: Trained Word2Vec model for word similarity.
            difficulty_threshold (float): Threshold above which words are considered difficult.
        
        Returns:
            str: Simplified sentence.
        """
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        simplified_words = []
        changed_words = []

        for word in words:

            vector = self.vectorizer.transform([word])
            if word not in self.X:
                difficulty = self.regressor.predict(vector)[0]  
            else: 
                difficulty = self.data[self.data['word']==word]['rating'].to_numpy()
            if difficulty > difficulty_threshold:
                try:
                    similar_words = word2vec_model.most_similar(word, topn=15)
                    possibilities = []
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector)[0]

                        if sim_difficulty <= difficulty:
                            possibilities.append((sim_word, sim_difficulty))
                            simplified_words.append(sim_word) 
                            changed_words.append((word, sim_word))
                            break
                        else:
                            simplified_words.append(word)
                            break
                
                except KeyError:
                    simplified_words.append(word)
            else:
                simplified_words.append(word)

        simplified_sentence = " ".join(simplified_words)
        return simplified_sentence, changed_words


class CWID_Prob:
    def __init__(self) -> None:
        print("making 2")

        self.wikipedia_train = pd.read_csv('../../CWID_train/Wikipedia_Train.csv')
        self.news_train = pd.read_csv('../../CWID_train/News_Train.csv')

        self.wiki_data = self.wikipedia_train.dropna(subset=["target_word", "probabilistic"]) 
        self.news_data = self.news_train.dropna(subset=["target_word", "probabilistic"]) 

        combined_df = pd.concat([self.wiki_data, self.news_data])

        # Retain rows with the largest 'Value' for each unique 'Key'
        self.data = combined_df.sort_values('probabilistic', ascending=False).drop_duplicates(subset='target_word', keep='first')

        # Reset index for a cleaner look
        self.data = self.data.reset_index(drop=True)


        self.X = self.data["target_word"].values  # Target words
        self.y = self.data["probabilistic"].values.astype(float)
        self.vectorizer = TfidfVectorizer()
        self.X_vectors = self.vectorizer.fit_transform(self.X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_vectors, self.y, test_size=0.2, random_state=42)

        # Train regressor
        self.regressor = joblib.load('../../CWID_Prob_Regressor.joblib') 
        # sents = list(self.wiki_data['sentence'])+list(self.news_data['sentence'])
        # w2v_sentences = [word_tokenize(sent) for sent in sents]
        # self.model = Word2Vec(w2v_sentences, seed=0, vector_size=100, window=5, min_count=5, workers=4)
        # self.model.save("our_model.model")
        self.model = Word2Vec.load("../../our_model.model")

    def simplify_sentence(self, sentence, difficulty_threshold=.25):
        """
        Simplifies a sentence by replacing difficult words with simpler alternatives.
        
        Args:
            sentence (str): Input sentence to be simplified.
            regressor: Trained regressor model for predicting word difficulty.
            vectorizer: Trained vectorizer for transforming words into features.
            word2vec_model: Trained Word2Vec model for word similarity.
            difficulty_threshold (float): Threshold above which words are considered difficult.
        
        Returns:
            str: Simplified sentence.
        """
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        simplified_words = []
        changed_words = []

        for word in words:

            vector = self.vectorizer.transform([word])
            if word not in self.X:
                difficulty = self.regressor.predict(vector)[0]  
            else: 
                difficulty = self.data[self.data['target_word']==word]['probabilistic'].to_numpy()
            if difficulty > difficulty_threshold:
                try:
                    similar_words = self.model.wv.most_similar(word, topn=20)
                    possibilities = []
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector)[0]

                        if sim_difficulty <= difficulty:
                            possibilities.append((sim_word, sim_difficulty))
                            # simplified_words.append(sim_word) 
                            # changed_words.append((word, sim_word))
                            break
                        # else:
                        #     simplified_words.append(word)
                    if possibilities:
                        lowest_diff = difficulty
                        best_word = word
                        for sim_word, sim_diff in possibilities: 
                            if sim_diff < lowest_diff:
                                lowest_diff = sim_diff
                                best_word = sim_word
                        simplified_words.append(best_word)
                        changed_words.append((word, best_word))
                    else:
                        simplified_words.append(word)
                
                except KeyError:
                    simplified_words.append(word)
            else:
                simplified_words.append(word)

        simplified_sentence = " ".join(simplified_words)
        return simplified_sentence, changed_words

class CWID_Bin:
    def __init__(self) -> None:
        print("making 3")

        self.wikipedia_train = pd.read_csv('../../CWID_train/Wikipedia_Train.csv')
        self.news_train = pd.read_csv('../../CWID_train/News_Train.csv')

        self.wiki_data = self.wikipedia_train.dropna(subset=["target_word", "binary"]) 
        self.news_data = self.news_train.dropna(subset=["target_word", "binary"]) 

        combined_df = pd.concat([self.wiki_data, self.news_data])

        # Retain rows with the largest 'Value' for each unique 'Key'
        self.data = combined_df.sort_values('binary', ascending=False).drop_duplicates(subset='target_word', keep='first')

        # Reset index for a cleaner look
        self.data = self.data.reset_index(drop=True)
        # sents = list(self.wiki_data['sentence'])+list(self.news_data['sentence'])
        # w2v_sentences = [word_tokenize(sent) for sent in sents]
        # self.model = Word2Vec(w2v_sentences, seed=0, vector_size=100, window=5, min_count=5, workers=4)
        self.model = Word2Vec.load("../../our_model.model")


        self.X = self.data["target_word"].values  # Target words
        self.y = self.data["binary"].values.astype(float)
        self.vectorizer = TfidfVectorizer()
        self.X_vectors = self.vectorizer.fit_transform(self.X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_vectors, self.y, test_size=0.2, random_state=42)

        # Train regressor
        self.regressor = joblib.load("../../CWID_Bin_Regressor.joblib")
    def simplify_sentence(self, sentence, difficulty_threshold=.5):
        """
        Simplifies a sentence by replacing difficult words with simpler alternatives.
        
        Args:
            sentence (str): Input sentence to be simplified.
            regressor: Trained regressor model for predicting word difficulty.
            vectorizer: Trained vectorizer for transforming words into features.
            word2vec_model: Trained Word2Vec model for word similarity.
            difficulty_threshold (float): Threshold above which words are considered difficult.
        
        Returns:
            str: Simplified sentence.
        """
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        simplified_words = []
        changed_words = []

        for word in words:
            vector = self.vectorizer.transform([word])
            if word not in self.X:
                difficulty = self.regressor.predict(vector)[0]  
            else: 
                difficulty = self.data[self.data['target_word']==word]['binary'].to_numpy()
            if difficulty > difficulty_threshold:
                try:
                    similar_words = self.model.wv.most_similar(word, topn=20)

                    possibilities = []
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector)[0]

                        if sim_difficulty <= difficulty:
                            possibilities.append((sim_word, sim_difficulty))
                            # simplified_words.append(sim_word) 
                            # changed_words.append((word, sim_word))
                            break
                        # else:
                        #     simplified_words.append(word)
                    if possibilities:
                        lowest_diff = difficulty
                        best_word = word
                        for sim_word, sim_diff in possibilities: 
                            if sim_diff < lowest_diff:
                                lowest_diff = sim_diff
                                best_word = sim_word
                        simplified_words.append(best_word)
                        changed_words.append((word, best_word))
                    else:
                        simplified_words.append(word)
                
                except KeyError:
                    simplified_words.append(word)
            else:
                simplified_words.append(word)

        simplified_sentence = " ".join(simplified_words)
        return simplified_sentence, changed_words

class CWID_Non_Native:
    def __init__(self) -> None:
        print("making 4")

        self.wikipedia_train = pd.read_csv('../../CWID_train/Wikipedia_Train.csv')
        self.news_train = pd.read_csv('../../CWID_train/News_Train.csv')

        self.wiki_data = self.wikipedia_train.dropna(subset=["target_word", "non_native_diff"]) 
        self.news_data = self.news_train.dropna(subset=["target_word", "non_native_diff"]) 

        combined_df = pd.concat([self.wiki_data, self.news_data])

        # Retain rows with the largest 'Value' for each unique 'Key'
        self.data = combined_df.sort_values('non_native_diff', ascending=False).drop_duplicates(subset='target_word', keep='first')

        # Reset index for a cleaner look
        self.data = self.data.reset_index(drop=True)


        self.X = self.data["target_word"].values  # Target words
        self.y = self.data["non_native_diff"].values.astype(float)
        self.vectorizer = TfidfVectorizer()
        self.X_vectors = self.vectorizer.fit_transform(self.X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_vectors, self.y, test_size=0.2, random_state=42)

        # Train regressor
        # self.regressor = joblib.load('CWID_Prob_Regressor.joblib') 
        # self.regressor = RandomForestClassifier()
        # self.regressor.fit(self.X_train, self.y_train)
        # joblib.dump(self.regressor, "CWID_NonNative_Regressor.joblib")
        self.regressor = joblib.load("../../CWID_NonNative_Regressor.joblib")
        # sents = list(self.wiki_data['sentence'])+list(self.news_data['sentence'])
        # w2v_sentences = [word_tokenize(sent) for sent in sents]
        # self.model = Word2Vec(w2v_sentences, seed=0, vector_size=100, window=5, min_count=5, workers=4)
        self.model = Word2Vec.load("../../our_model.model")

    def simplify_sentence(self, sentence, difficulty_threshold=.25):
        """
        Simplifies a sentence by replacing difficult words with simpler alternatives.
        
        Args:
            sentence (str): Input sentence to be simplified.
            regressor: Trained regressor model for predicting word difficulty.
            vectorizer: Trained vectorizer for transforming words into features.
            word2vec_model: Trained Word2Vec model for word similarity.
            difficulty_threshold (float): Threshold above which words are considered difficult.
        
        Returns:
            str: Simplified sentence.
        """
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        simplified_words = []
        changed_words = []

        for word in words:

            vector = self.vectorizer.transform([word])
            if word not in self.X:
                difficulty = self.regressor.predict(vector)[0]  
            else: 
                difficulty = self.data[self.data['target_word']==word]['non_native_diff'].to_numpy()
            if difficulty > difficulty_threshold:
                try:
                    similar_words = self.model.wv.most_similar(word, topn=20)
                    possibilities = []
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector)[0]

                        if sim_difficulty <= difficulty:
                            possibilities.append((sim_word, sim_difficulty))
                            # simplified_words.append(sim_word) 
                            # changed_words.append((word, sim_word))
                            break
                        # else:
                        #     simplified_words.append(word)
                    if possibilities:
                        lowest_diff = difficulty
                        best_word = word
                        for sim_word, sim_diff in possibilities: 
                            if sim_diff < lowest_diff:
                                lowest_diff = sim_diff
                                best_word = sim_word
                        simplified_words.append(best_word)
                        changed_words.append((word, best_word))
                    else:
                        simplified_words.append(word)
                
                except KeyError:
                    simplified_words.append(word)
            else:
                simplified_words.append(word)

        simplified_sentence = " ".join(simplified_words)
        return simplified_sentence, changed_words
    def simplify_sentence_pretrain(self, sentence, difficulty_threshold=.25):
        """
        Simplifies a sentence by replacing difficult words with simpler alternatives.
        
        Args:
            sentence (str): Input sentence to be simplified.
            regressor: Trained regressor model for predicting word difficulty.
            vectorizer: Trained vectorizer for transforming words into features.
            word2vec_model: Trained Word2Vec model for word similarity.
            difficulty_threshold (float): Threshold above which words are considered difficult.
        
        Returns:
            str: Simplified sentence.
        """
        # Tokenize the sentence into words
        words = nltk.word_tokenize(sentence)
        simplified_words = []
        changed_words = []

        for word in words:

            vector = self.vectorizer.transform([word])
            if word not in self.X:
                difficulty = self.regressor.predict(vector)[0]  
            else: 
                difficulty = self.data[self.data['target_word']==word]['non_native_diff'].to_numpy()
            if difficulty > difficulty_threshold:
                try:
                    similar_words = word2vec_model.most_similar(word, topn=20)
                    possibilities = []
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector)[0]

                        if sim_difficulty <= difficulty:
                            possibilities.append((sim_word, sim_difficulty))
                            # simplified_words.append(sim_word) 
                            # changed_words.append((word, sim_word))
                            break
                        # else:
                        #     simplified_words.append(word)
                    if possibilities:
                        lowest_diff = difficulty
                        best_word = word
                        for sim_word, sim_diff in possibilities: 
                            if sim_diff < lowest_diff:
                                lowest_diff = sim_diff
                                best_word = sim_word
                        simplified_words.append(best_word)
                        changed_words.append((word, best_word))
                    else:
                        simplified_words.append(word)
                
                except KeyError:
                    simplified_words.append(word)
            else:
                simplified_words.append(word)

        simplified_sentence = " ".join(simplified_words)
        return simplified_sentence, changed_words

def create_parser():
    parser = argparse.ArgumentParser(description='Given a sentence, pick out difficult words and simplify them')
    parser.add_argument('--sentence', type=str, required=True,
                        help='sentence to simplify')

    return parser

def get_sentences(sentence):
    print("gen models")
    wcl_model = WCL()
    prob_model = CWID_Prob()
    bin_model = CWID_Bin()
    nonnative_model = CWID_Non_Native()
    results = {}

    print("making sentences")
    wcl_simp, wcl_changed = wcl_model.simplify_sentence(sentence, difficulty_threshold=2.5)
    results["WCL/pretrained"] = (wcl_simp, wcl_changed)

    prob_simp, prob_changed = prob_model.simplify_sentence(sentence, difficulty_threshold=.2)
    results["CWID-probabilistic/ours"] = (prob_simp, prob_changed)

    bin_simp, bin_changed = bin_model.simplify_sentence(sentence, difficulty_threshold=.5)
    results["CWID-binary/ours"] = (bin_simp, bin_changed)

    nonnative_simp, nonnative_changed = nonnative_model.simplify_sentence(sentence, difficulty_threshold=3)
    results["CWID-nonnative/ours"] = (nonnative_simp, nonnative_changed)

    nonnative_simp2, nonnative_changed2 = nonnative_model.simplify_sentence_pretrain(sentence, difficulty_threshold=3)
    results["CWID-nonnative/pretrained"] = (nonnative_simp2, nonnative_changed2)

    return results

class WCL_Testing:
    def __init__(self):
        print("making 1")
        self.lexicon_data = pd.read_csv('../../WCL_data/lexicon.csv')
        self.data = self.lexicon_data.dropna(subset=["word", "rating"]) 
        self.X = self.data["word"].values  # Target words
        self.y = self.data["rating"].values.astype(float)
        self.vectorizer = TfidfVectorizer()
        self.X_vectors = self.vectorizer.fit_transform(self.X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_vectors, self.y, test_size=0.2, random_state=42)

        # Train regressor
        self.regressor = self.nn()


    def simplify_sentence(self, sentence, difficulty_threshold=3):
        words = nltk.word_tokenize(sentence)
        simplified_words = []
        changed_words = []

        for word in words:
            vector = self.vectorizer.transform([word])

            # Handle word not in vocabulary
            if vector.shape[0] == 0:  # No features for word
                print(f"Word '{word}' not in vocabulary, assigning default difficulty.")
                difficulty = 0
            else:
                difficulty = self.regressor.predict(vector.toarray(), verbose=0)[0]  # Convert sparse to dense

            # If word difficulty is above threshold, try to replace it with simpler words
            if difficulty > difficulty_threshold:
                try:
                    similar_words = word2vec_model.most_similar(word, topn=15)
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector.toarray(), verbose=0)[0]

                        if sim_difficulty <= difficulty:
                            simplified_words.append(sim_word)
                            changed_words.append((word, sim_word))
                            break  # Replace with the first valid simplification
                    else:
                        simplified_words.append(word)  # Keep original if no valid simpler word
                except KeyError:
                    # If the word is not in Word2Vec, keep it as is
                    simplified_words.append(word)
            else:
                simplified_words.append(word)

        simplified_sentence = " ".join(simplified_words)
        return simplified_sentence, changed_words

    def random_forest(self):
        self.regressor = joblib.load("../../WCL_regressor.pkl")
        y_pred = self.regressor.predict(self.X_test)
        print("Mean Squared Error:", mean_squared_error(self.y_test, y_pred))
        print("R-squared Score:", r2_score(self.y_test, y_pred))

    def nn(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),  # Input layer
            Dense(32, activation='relu'),  # Hidden layer
            Dense(1, activation='linear')  # Output layer for regression
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        # Train the model
        model.fit(self.X_train, self.y_train, epochs=25, batch_size=32, verbose=0)
        # Evaluate the model
        # loss, mae = model.evaluate(self.X_test, self.y_test)
        # y_pred = model.predict(self.X_test, verbose=0)
        # print("Mean Squared Error:", mean_squared_error(self.y_test, y_pred))
        # print("R-squared Score:", r2_score(self.y_test, y_pred))
        return model

class CWID_Prob_Testing:
    def __init__(self) -> None:
        print("making 2")

        self.wikipedia_train = pd.read_csv('../../CWID_train/Wikipedia_Train.csv')
        self.news_train = pd.read_csv('../../CWID_train/News_Train.csv')

        self.wiki_data = self.wikipedia_train.dropna(subset=["target_word", "probabilistic"]) 
        self.news_data = self.news_train.dropna(subset=["target_word", "probabilistic"]) 

        combined_df = pd.concat([self.wiki_data, self.news_data])

        # Retain rows with the largest 'Value' for each unique 'Key'
        self.data = combined_df.sort_values('probabilistic', ascending=False).drop_duplicates(subset='target_word', keep='first')

        # Reset index for a cleaner look
        self.data = self.data.reset_index(drop=True)


        self.X = self.data["target_word"].values  # Target words
        self.y = self.data["probabilistic"].values.astype(float)
        self.vectorizer = TfidfVectorizer()
        self.X_vectors = self.vectorizer.fit_transform(self.X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_vectors, self.y, test_size=0.2, random_state=42)
        self.model = Word2Vec.load("../../our_model.model")
        self.regressor = joblib.load('../../CWID_Prob_Regressor.joblib') 
    def simplify_sentence(self, sentence, difficulty_threshold=3):
        words = nltk.word_tokenize(sentence)
        simplified_words = []
        changed_words = []

        for word in words:
            vector = self.vectorizer.transform([word])

            # Handle word not in vocabulary
            if vector.shape[0] == 0:  # No features for word
                print(f"Word '{word}' not in vocabulary, assigning default difficulty.")
                difficulty = 0
            else:
                difficulty = self.regressor.predict(vector.toarray())[0]  # Convert sparse to dense

            # If word difficulty is above threshold, try to replace it with simpler words
            if difficulty > difficulty_threshold:
                try:
                    similar_words = word2vec_model.wv.most_similar(word, topn=15)
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector.toarray())[0]

                        if sim_difficulty <= difficulty:
                            simplified_words.append(sim_word)
                            changed_words.append((word, sim_word))
                            break  # Replace with the first valid simplification
                    else:
                        simplified_words.append(word)  # Keep original if no valid simpler word
                except KeyError:
                    # If the word is not in Word2Vec, keep it as is
                    simplified_words.append(word)
            else:
                simplified_words.append(word)

        simplified_sentence = " ".join(simplified_words)
        return simplified_sentence, changed_words

    def random_forest(self):
        self.regressor = joblib.load('../../CWID_Prob_Regressor.joblib') 
        y_pred = self.regressor.predict(self.X_test)
        print("Mean Squared Error:", mean_squared_error(self.y_test, y_pred))
        print("R-squared Score:", r2_score(self.y_test, y_pred))

    def nn(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),  # Input layer
            Dense(32, activation='relu'),  # Hidden layer
            Dense(1, activation='linear')  # Output layer for regression
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        # Train the model
        model.fit(self.X_train, self.y_train, epochs=25, batch_size=32, verbose=0)
        # Evaluate the model
        # loss, mae = model.evaluate(self.X_test, self.y_test)
        y_pred = model.predict(self.X_test, verbose=0)
        print("Mean Squared Error:", mean_squared_error(self.y_test, y_pred))
        print("R-squared Score:", r2_score(self.y_test, y_pred))
        return model

class CWID_Non_Native_Testing:
    def __init__(self) -> None:
        print("making 4")

        self.wikipedia_train = pd.read_csv('../../CWID_train/Wikipedia_Train.csv')
        self.news_train = pd.read_csv('../../CWID_train/News_Train.csv')

        self.wiki_data = self.wikipedia_train.dropna(subset=["target_word", "non_native_diff"]) 
        self.news_data = self.news_train.dropna(subset=["target_word", "non_native_diff"]) 

        combined_df = pd.concat([self.wiki_data, self.news_data])

        # Retain rows with the largest 'Value' for each unique 'Key'
        self.data = combined_df.sort_values('non_native_diff', ascending=False).drop_duplicates(subset='target_word', keep='first')

        # Reset index for a cleaner look
        self.data = self.data.reset_index(drop=True)


        self.X = self.data["target_word"].values  # Target words
        self.y = self.data["non_native_diff"].values.astype(float)
        self.vectorizer = TfidfVectorizer()
        self.X_vectors = self.vectorizer.fit_transform(self.X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_vectors, self.y, test_size=0.2, random_state=42)

        # Train regressor

        # self.regressor = joblib.load("../../CWID_NonNative_Regressor.joblib")

        # self.model = Word2Vec.load("../../our_model.model")
    def simplify_sentence(self, sentence, difficulty_threshold=3):
        words = nltk.word_tokenize(sentence)
        simplified_words = []
        changed_words = []

        for word in words:
            vector = self.vectorizer.transform([word])

            # Handle word not in vocabulary
            if vector.shape[0] == 0:  # No features for word
                print(f"Word '{word}' not in vocabulary, assigning default difficulty.")
                difficulty = 0
            else:
                difficulty = self.regressor.predict(vector.toarray())[0]  # Convert sparse to dense

            # If word difficulty is above threshold, try to replace it with simpler words
            if difficulty > difficulty_threshold:
                try:
                    similar_words = word2vec_model.most_similar(word, topn=15)
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector.toarray())[0]

                        if sim_difficulty <= difficulty:
                            simplified_words.append(sim_word)
                            changed_words.append((word, sim_word))
                            break  # Replace with the first valid simplification
                    else:
                        simplified_words.append(word)  # Keep original if no valid simpler word
                except KeyError:
                    # If the word is not in Word2Vec, keep it as is
                    simplified_words.append(word)
            else:
                simplified_words.append(word)

        simplified_sentence = " ".join(simplified_words)
        return simplified_sentence, changed_words

    def random_forest(self):
        self.regressor = joblib.load('../../CWID_Prob_Regressor.joblib') 
        y_pred = self.regressor.predict(self.X_test)
        print("Mean Squared Error:", mean_squared_error(self.y_test, y_pred))
        print("R-squared Score:", r2_score(self.y_test, y_pred))

    def nn(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),  # Input layer
            Dense(32, activation='relu'),  # Hidden layer
            Dense(1, activation='linear')  # Output layer for regression
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

        # Train the model
        model.fit(self.X_train, self.y_train, epochs=25, batch_size=32, verbose=0)
        # Evaluate the model
        # loss, mae = model.evaluate(self.X_test, self.y_test)
        y_pred = model.predict(self.X_test, verbose=0)
        print("Mean Squared Error:", mean_squared_error(self.y_test, y_pred))
        print("R-squared Score:", r2_score(self.y_test, y_pred))
        # return model



def test_regessor_classifier():
    print('WCL')
    wcl_test = WCL_Testing()
    wcl_test.random_forest()
    wcl_test.nn()
    print('CWID')
    cwid_test = CWID_Prob_Testing()
    cwid_test.random_forest()
    cwid_test.nn()
    print('CWID NonN')
    cwidnonn_test = CWID_Non_Native_Testing()
    cwidnonn_test.random_forest()
    cwidnonn_test.nn()
    pass

if __name__ == '__main__':
    parser = create_parser()

    args = vars(parser.parse_args())
    sentence = args['sentence']
    # test_regessor_classifier()
    # print(get_sentences(sentence))
    # test_sentences = [
    #     "The obfuscation of the report's findings was intentional, aiming to confound any cursory reader.",
    #     "Despite his ostensible altruism, his ulterior motives became glaringly evident over time.",
    #     "The juxtaposition of the protagonist's arcane motivations against the antagonist's overt simplicity was striking.",
    #     "accumulated, thesaurus, differing, terror"
    # ]


    wcl_model = WCL_Testing()
    prob_model = CWID_Prob()
    bin_model = CWID_Bin()
    nonnative_model = CWID_Non_Native()
    print("=" * 100)
    print("INPUT:")
    print("=" * 100)
    print("Original Sentence:", sentence, '\n')

    print("=" * 100)
    print("WCL DATA: 1 - 6: similarity based on pretrained model")
    print("=" * 100)
    wcl_simp, wcl_changed = wcl_model.simplify_sentence(sentence, difficulty_threshold=2.5)
    scores = scorer.score(sentence, wcl_simp)
    for key in scores:
        print(f'{key}: {scores[key]}')
    print("-" * 50)
    print("Simplified Sentence:", wcl_simp)
    print("Words Changed:", wcl_changed, "\n")

    print("=" * 100)
    print("CWID PROB DATA: .1 - 1: similarity based on sentences from this dataset")
    print("=" * 100)
    prob_simp, prob_changed = prob_model.simplify_sentence(sentence, difficulty_threshold=.2)
    scores = scorer.score(sentence, prob_simp)
    for key in scores:
        print(f'{key}: {scores[key]}')
    print("Simplified Sentence:", prob_simp)
    print("Words Changed:", prob_changed, "\n")

    # print("=" * 100)
    # print("CWID BINARY DATA: 0/1: similarity based on sentences from this dataset")
    # print("=" * 100)
    # bin_simp, bin_changed = bin_model.simplify_sentence(sentence, difficulty_threshold=.5)
    # scores = scorer.score(sentence, bin_simp)
    # for key in scores:
    #     print(f'{key}: {scores[key]}')
    # print("Simplified Sentence:", bin_simp)
    # print("Words Changed:", bin_changed, "\n")

    print("=" * 100)
    print("CWID NON-NATIVE DATA: 0 - 10: similarity based on sentences from this dataset")
    print("=" * 100)
    nonnative_simp, nonnative_changed = nonnative_model.simplify_sentence(sentence, difficulty_threshold=3)
    scores = scorer.score(sentence, nonnative_simp)
    for key in scores:
        print(f'{key}: {scores[key]}')
    print("Simplified Sentence:", nonnative_simp)
    print("Words Changed:", nonnative_changed, "\n")

    print("=" * 100)
    print("CWID NON-NATIVE DATA: 0 - 10: similarity based on pretrained model")
    print("=" * 100)
    nonnative_simp2, nonnative_changed2 = nonnative_model.simplify_sentence_pretrain(sentence, difficulty_threshold=3)
    scores = scorer.score(sentence, nonnative_simp2)
    for key in scores:
        print(f'{key}: {scores[key]}')
    print("Simplified Sentence:", nonnative_simp2)
    print("Words Changed:", nonnative_changed2, "\n")