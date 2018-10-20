import string
import re
from stop_words import get_stop_words
from stemming.porter2 import stem

class TextHelper():

    def __init__(self):
        self.stop_words = get_stop_words('en')

    def clean(self, s):
        translator = "".maketrans("", "", string.punctuation)
        return s.translate(translator)

    def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split("\W+", text)

    def remove_stopwords(self, word_tokens):
  
        filtered_sentence = [w for w in word_tokens if not w in self.stop_words] 

        return filtered_sentence

    def remove_numbers_short_words(self, word_tokens):

        filtered_sentence = [w for w in word_tokens if ((not w.isdigit()) and len(w)>1)] 

        return filtered_sentence

    def stem_words(self, word_tokens):

        return [stem(word) for word in word_tokens] 

    def tokenize_and_clean(self, text):

        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.remove_numbers_short_words(tokens)
        tokens = self.stem_words(tokens)

        return tokens