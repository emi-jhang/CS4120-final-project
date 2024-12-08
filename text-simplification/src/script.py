# IMPORTS
print("Importing")
import warnings
warnings.filterwarnings("ignore")

import argparse
import glob
import os
import os.path as osp
import json
from langdetect import detect

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download('punkt')
from nltk.corpus import cmudict
nltk.download('cmudict')
d = cmudict.dict()
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import joblib

import re
from rouge_score import rouge_scorer
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from transformers import T5ForConditionalGeneration, T5Tokenizer

# GLOBAL VARIABLES 
import gensim.downloader as api

model_path = "../../GoogleNews-vectors-negative300-SLIM.bin.gz"
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
our_model = Word2Vec.load("../../our_model.model")
print("vars")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def syllable_count(word):
    """Return the syllable count for a word."""
    word = word.lower()
    if word in d:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in d[word]])  # Get the max syllables
    else:
        return None

def flesch_kincaid(text):
    """
    Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level for a given text.
    """
    if text.strip() == '' or detect(text) != 'en':
        return -1
    
    # Split text into sentences
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings
    num_sentences = len(sentences)

    # Split text into words
    words = re.findall(r'\w+', text)
    num_words = len(words)

    # Count syllables in words
    num_syllables = sum(syllable_count(word) for word in words if bool(re.fullmatch(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", word)) == True)

    # Calculate ASL and ASW
    asl = num_words / num_sentences if num_sentences > 0 else 0
    asw = num_syllables / num_words if num_words > 0 else 0

    # Calculate Flesch Reading Ease
    reading_ease = 206.835 - (1.015 * asl) - (84.6 * asw)

    return reading_ease

class WCL:
    def __init__(self):
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

        # Get difficulty of each word in the sentence from WCL dataset or, if not in dataset, our WCL regressor, and if above a certain threshold, replace word using word2vec
        for word in words:
            vector = self.vectorizer.transform([word])
            if word not in self.X:
                difficulty = self.regressor.predict(vector)[0]  
            else: 
                difficulty = self.data[self.data['word']==word]['rating'].to_numpy()
            if difficulty > difficulty_threshold:
                try:
                    # Get list of similar words from word2vec and append simpler word or, if none found, original work
                    similar_words = word2vec_model.most_similar(word, topn=15)
                    # Loop through similar words and only use if difficulty is lower so that we get the lowest difficulty word to swap in
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector)[0]

                        if sim_difficulty < difficulty and sim_word != word:
                            simplified_words.append(sim_word)
                            break
                        else: 
                            simplified_words.append(word)
                
                except KeyError:
                    simplified_words.append(word)
            else:
                simplified_words.append(word)

        simplified_sentence = " ".join(simplified_words)
        print(simplified_sentence)
        return simplified_sentence, changed_words


class CWID_Prob:
    def __init__(self) -> None:
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
        self.model = our_model

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

        # Get difficulty of each word in the sentence from CWID dataset or, if not in dataset, our CWID regressor, and if above a certain threshold, replace word using word2vec
        for word in words:
            vector = self.vectorizer.transform([word])
            if word not in self.X:
                difficulty = self.regressor.predict(vector)[0]  
            else: 
                difficulty = self.data[self.data['target_word']==word]['probabilistic'].to_numpy()
            if difficulty > difficulty_threshold:
                # Get list of similar words from word2vec and append simpler word or, if none found, original work
                try:
                    # Get list of similar words from word2vec and append simpler word or, if none found, original work
                    similar_words = word2vec_model.most_similar(word, topn=15)
                    # Loop through similar words and only use if difficulty is lower so that we get the lowest difficulty word to swap in
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector)[0]

                        if sim_difficulty < difficulty and sim_word != word:
                            simplified_words.append(sim_word)
                            break
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
        self.wikipedia_train = pd.read_csv('../../CWID_train/Wikipedia_Train.csv')
        self.news_train = pd.read_csv('../../CWID_train/News_Train.csv')

        self.wiki_data = self.wikipedia_train.dropna(subset=["target_word", "binary"]) 
        self.news_data = self.news_train.dropna(subset=["target_word", "binary"]) 

        combined_df = pd.concat([self.wiki_data, self.news_data])

        # Retain rows with the largest 'Value' for each unique 'Key'
        self.data = combined_df.sort_values('binary', ascending=False).drop_duplicates(subset='target_word', keep='first')

        # Reset index for a cleaner look
        self.data = self.data.reset_index(drop=True)
        self.model = our_model


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

        # Get difficulty of each word in the sentence from CWID dataset or, if not in dataset, our CWID regressor, and if above a certain threshold, replace word using word2vec
        for word in words:
            vector = self.vectorizer.transform([word])
            if word not in self.X:
                difficulty = self.regressor.predict(vector)[0]  
            else: 
                difficulty = self.data[self.data['target_word']==word]['binary'].to_numpy()
            if difficulty > difficulty_threshold:
                # Get list of similar words from word2vec and append simpler word or, if none found, original work
                try:
                    # Get list of similar words from word2vec and append simpler word or, if none found, original work
                    similar_words = word2vec_model.most_similar(word, topn=15)
                    # Loop through similar words and only use if difficulty is lower so that we get the lowest difficulty word to swap in
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector)[0]

                        if sim_difficulty < difficulty and sim_word != word:
                            simplified_words.append(sim_word)
                            break
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
        self.regressor = joblib.load("../../CWID_NonNative_Regressor.joblib")
        self.model = our_model
        self.pretrained_model = word2vec_model

    def simplify_sentence(self, sentence, difficulty_threshold=.25, pretrained=False):
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

        # Get difficulty of each word in the sentence from CWID dataset or, if not in dataset, our CWID regressor, and if above a certain threshold, replace word using word2vec
        for word in words:

            vector = self.vectorizer.transform([word])
            if word not in self.X:
                difficulty = self.regressor.predict(vector)[0]  
            else: 
                difficulty = self.data[self.data['target_word']==word]['non_native_diff'].to_numpy()
            if difficulty > difficulty_threshold:
                # Get list of similar words from word2vec and append simpler word or, if none found, original work
                try:
                    # Get list of similar words from word2vec and append simpler word or, if none found, original work
                    similar_words = word2vec_model.most_similar(word, topn=15)
                    # Loop through similar words and only use if difficulty is lower so that we get the lowest difficulty word to swap in
                    for sim_word, word_sim in similar_words:
                        sim_vector = self.vectorizer.transform([sim_word])
                        sim_difficulty = self.regressor.predict(sim_vector)[0]

                        if sim_difficulty < difficulty and sim_word != word:
                            simplified_words.append(sim_word)
                            break
                        else: 
                            simplified_words.append(word)
                
                except KeyError:
                    simplified_words.append(word)
            else:
                simplified_words.append(word)

        simplified_sentence = " ".join(simplified_words)
        return simplified_sentence, changed_words

class T5_Model:
    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained('../../t5_model')
        self.tokenizer = T5Tokenizer.from_pretrained('../../t5_tokenizer')

    def simplify_sentence(self, sentence):
        """
        Simplifies a sentence by plugging it into a T5 model pretrained on a text simplification dataset. 
        
        Args:
            sentence (str): Input sentence to be simplified.
        
        Returns:
            str: Simplified sentence.
        """
        # Tokenize the input text
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        
        # Generate the simplified sentence
        outputs = self.model.generate(inputs["input_ids"], max_length=50, num_beams=4, early_stopping=True)
        
        # Decode the generated output to get the simplified text
        simplified_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return simplified_text


def create_parser():
    parser = argparse.ArgumentParser(description='Given a sentence, pick out difficult words and simplify them')
    parser.add_argument('--sentence', type=str, required=True,
                        help='sentence to simplify')

    return parser

def get_sentences(sentence):
    wcl_model = WCL()
    prob_model = CWID_Prob()
    bin_model = CWID_Bin()
    nonnative_model = CWID_Non_Native()
    t5_model = T5_model()
    results = {}

    wcl_simp, wcl_changed = wcl_model.simplify_sentence(sentence, difficulty_threshold=2.5)
    results["WCL/pretrained"] = (wcl_simp, wcl_changed)

    prob_simp, prob_changed = prob_model.simplify_sentence(sentence, difficulty_threshold=.2)
    results["CWID-probabilistic/ours"] = (prob_simp, prob_changed)

    bin_simp, bin_changed = bin_model.simplify_sentence(sentence, difficulty_threshold=.5)
    results["CWID-binary/ours"] = (bin_simp, bin_changed)

    nonnative_simp, nonnative_changed = nonnative_model.simplify_sentence(sentence, difficulty_threshold=3)
    results["CWID-nonnative/ours"] = (nonnative_simp, nonnative_changed)

    nonnative_simp2, nonnative_changed2 = nonnative_model.simplify_sentence(sentence, difficulty_threshold=3, pretrained=True)
    results["CWID-nonnative/pretrained"] = (nonnative_simp2, nonnative_changed2)

    t5_model.simplify_sentence(sentence)
    results["T5"] = sentence

    return results

if __name__ == '__main__':
    parser = create_parser()

    args = vars(parser.parse_args())
    sentence = args['sentence']

    wcl_model = WCL()
    prob_model = CWID_Prob()
    bin_model = CWID_Bin()
    nonnative_model = CWID_Non_Native()
    t5_model = T5_Model()

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
    print("Original Sentence Reading Ease:", flesch_kincaid(sentence))
    print("Simplified Sentence Reading Ease:", flesch_kincaid(wcl_simp))
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
    print("Original Sentence Reading Ease:", flesch_kincaid(sentence))
    print("Simplified Sentence Reading Ease:", flesch_kincaid(prob_simp))
    print("-" * 50)
    print("Simplified Sentence:", prob_simp)
    print("Words Changed:", prob_changed, "\n")

    print("=" * 100)
    print("CWID NON-NATIVE DATA: 0 - 10: similarity based on sentences from this dataset")
    print("=" * 100)
    nonnative_simp, nonnative_changed = nonnative_model.simplify_sentence(sentence, difficulty_threshold=3)
    scores = scorer.score(sentence, nonnative_simp)
    for key in scores:
        print(f'{key}: {scores[key]}')
    print("Original Sentence Reading Ease:", flesch_kincaid(sentence))
    print("Simplified Sentence Reading Ease:", flesch_kincaid(nonnative_simp))
    print("-" * 50)
    print("Simplified Sentence:", nonnative_simp)
    print("Words Changed:", nonnative_changed, "\n")

    print("=" * 100)
    print("CWID NON-NATIVE DATA: 0 - 10: similarity based on pretrained model")
    print("=" * 100)
    nonnative_simp2, nonnative_changed2 = nonnative_model.simplify_sentence(sentence, difficulty_threshold=3, pretrained=True)
    scores = scorer.score(sentence, nonnative_simp2)
    for key in scores:
        print(f'{key}: {scores[key]}')
    print("Original Sentence Reading Ease:", flesch_kincaid(sentence))
    print("Simplified Sentence Reading Ease:", flesch_kincaid(nonnative_simp2))
    print("-" * 50)
    print("Simplified Sentence:", nonnative_simp2)
    print("Words Changed:", nonnative_changed2, "\n")

    print("=" * 100)
    print("T5:")
    print("=" * 100)
    t5_simp = t5_model.simplify_sentence(sentence)
    scores = scorer.score(sentence, t5_simp)
    for key in scores:
        print(f'{key}: {scores[key]}')
    print("Original Sentence Reading Ease:", flesch_kincaid(sentence))
    print("Simplified Sentence Reading Ease:", flesch_kincaid(t5_simp))
    print("-" * 50)
    print("Simplified Sentence:", t5_simp)