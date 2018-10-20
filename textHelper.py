"""
SCC231 - Introdução a Sistemas Inteligentes
Professor: Alneu de Andrade Lopes
PAE: Fabiana Góes


Projeto - Classificação textual com Naive Bayes

Aluno: Elisa Saltori Trujillo (8551100)


"""

import string
import re
from stop_words import get_stop_words
from stemming.porter2 import stem

class TextHelper():
    """
    Class has methods for pre-processing text, such as tokenizing and stemming

    """
    def __init__(self):
        self.stop_words = get_stop_words('en')

    def clean(self, s):
        translator = "".maketrans("", "", string.punctuation)
        return s.translate(translator)

    def tokenize(self, text):
        """
        Gets text and turns it into an array of words

        Parameters:
            text - input text

        """
        text = self.clean(text).lower()
        return re.split("\W+", text)

    def remove_stopwords(self, word_tokens):
        """
        Remove stop words from array of words

        Parameters:
            word_tokens: array with words representing a text
        """
  
        filtered_sentence = [w for w in word_tokens if not w in self.stop_words] 

        return filtered_sentence

    def remove_numbers_short_words(self, word_tokens):
        """
        Remove numbers and words of a single character from array of words

        Parameters:
            word_tokens: array with words representing a text

        """

        filtered_sentence = [w for w in word_tokens if ((not w.isdigit()) and len(w)>1)] 

        return filtered_sentence

    def stem_words(self, word_tokens):
        """
        Stem words in words array. Uses stem function from stemming module

        Parameters:
            word_tokens: array with words representing a text
        
        """
        return [stem(word) for word in word_tokens] 


    def tokenize_and_clean(self, text):
        """
        Apply all pre-processing functions:
            -tokenizing
            -removing stop words
            -removing numbers and single letter words
            -stems words

        """
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.remove_numbers_short_words(tokens)
        tokens = self.stem_words(tokens)

        return tokens