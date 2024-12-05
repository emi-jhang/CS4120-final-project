# IMPORTS
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

# GLOBAL VARIABLES 
import gensim.downloader as api
word2vec_model = api.load('word2vec-google-news-300')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

class WCL:
    def __init__(self):
        self.lexicon_data = pd.read_csv('WCL_data/lexicon.csv')
        self.data = self.lexicon_data.dropna(subset=["word", "rating"]) 
        self.X = self.data["word"].values  # Target words
        self.y = self.data["rating"].values.astype(float)
        self.vectorizer = TfidfVectorizer()
        self.X_vectors = self.vectorizer.fit_transform(self.X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_vectors, self.y, test_size=0.2, random_state=42)

        # Train regressor
        self.regressor = joblib.load("WCL_regressor.pkl")


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
                    similar_words = word2vec_model.most_similar(word, topn=5)
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


class CWID_Prob:
    def __init__(self) -> None:
        self.wikipedia_train = pd.read_csv('CWID_train/Wikipedia_Train.csv')
        self.news_train = pd.read_csv('CWID_train/News_Train.csv')

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
        self.regressor = joblib.load('CWID_Prob_Regressor.joblib') 
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
                    similar_words = word2vec_model.most_similar(word, topn=50)
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
        self.wikipedia_train = pd.read_csv('CWID_train/Wikipedia_Train.csv')
        self.news_train = pd.read_csv('CWID_train/News_Train.csv')

        self.wiki_data = self.wikipedia_train.dropna(subset=["target_word", "binary"]) 
        self.news_data = self.news_train.dropna(subset=["target_word", "binary"]) 

        combined_df = pd.concat([self.wiki_data, self.news_data])

        # Retain rows with the largest 'Value' for each unique 'Key'
        self.data = combined_df.sort_values('binary', ascending=False).drop_duplicates(subset='target_word', keep='first')

        # Reset index for a cleaner look
        self.data = self.data.reset_index(drop=True)


        self.X = self.data["target_word"].values  # Target words
        self.y = self.data["binary"].values.astype(float)
        self.vectorizer = TfidfVectorizer()
        self.X_vectors = self.vectorizer.fit_transform(self.X)

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_vectors, self.y, test_size=0.2, random_state=42)

        # Train regressor
        self.regressor = joblib.load("CWID_Bin_Regressor.joblib")
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
                    similar_words = word2vec_model.most_similar(word, topn=50)
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
    parser.add_argument('--sentences', type=str, required=True, nargs="*",
                        help='threshold for the word difficulty ()')

    return parser


if __name__ == '__main__':
    parser = create_parser()

    # args = vars(parser.parse_args())

    # threshold = args['threshold']
    test_sentences = [
        "This is a simple sentence.",
        "Although she was considered smart, she failed all her exams.",
        "Anachronism in historical contexts can be confusing.",
        "accumulated, thesaurus, differing, terror"
    ]
    print("=" * 100)
    print("WCL DATA:")
    print("=" * 100)

    wcl_model = WCL()

    for sentence in test_sentences:
        simplified, changed_words = wcl_model.simplify_sentence(
            sentence, 
            difficulty_threshold=2
        )
        print("Original Sentence:", sentence)
        print("Simplified Sentence:", simplified)
        print("Words Changed:", changed_words, "\n")

        scores = scorer.score(sentence, simplified)
        for key in scores:
            print(f'{key}: {scores[key]}')
        print("-" * 50)
    print("=" * 100)
    print("CWID PROBABILISTIC DATA:")
    print("=" * 100)

    prob_model = CWID_Prob()

    for sentence in test_sentences:
        simplified, changed_words = prob_model.simplify_sentence(
            sentence, 
            difficulty_threshold=.1
        )
        print("Original Sentence:", sentence)
        print("Simplified Sentence:", simplified)
        print("Words Changed:", changed_words, "\n")

        scores = scorer.score(sentence, simplified)
        for key in scores:
            print(f'{key}: {scores[key]}')
        print("-" * 50)

    print("=" * 100)
    print("CWID BINARY DATA:")
    print("=" * 100)

    bin_model = CWID_Bin()

    for sentence in test_sentences:
        simplified, changed_words = bin_model.simplify_sentence(
            sentence, 
            difficulty_threshold=.5
        )
        print("Original Sentence:", sentence)
        print("Simplified Sentence:", simplified)
        print("Words Changed:", changed_words, "\n")

        scores = scorer.score(sentence, simplified)
        for key in scores:
            print(f'{key}: {scores[key]}')
        print("-" * 50)