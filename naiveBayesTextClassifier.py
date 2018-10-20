"""
SCC231 - Introdução a Sistemas Inteligentes
Professor: Alneu de Andrade Lopes
PAE: Fabiana Góes


Projeto - Classificação textual com Naive Bayes

Aluno: Elisa Saltori Trujillo (8551100)


"""

from collections import Counter
from textHelper import TextHelper
import math

class NaiveBayesTextClassifier():
    """
    Implementation of naive bayes classifier for classifying texts

    """
    def __init__(self):
        self.helper = TextHelper()


    def fit(self, data, targets):
        """
        Train model with the given data

        Parameters:
            data - array with data. Each position should be a string with the text of a document.
            target - target class of the data
    
        """

        self.vocab = set() #global vocabulary
        self.class_count  = Counter(targets) #number of instances of each class
        self.class_prob = {} #probability of each class
        
        #class probability
        #use log of probability so classifier doesnt underflow
        for c in self.class_count:
            self.class_prob[c] = float(self.class_count[c])/len(targets) 
            self.class_prob[c] = math.log(self.class_prob[c])
        
        self.word_counts = {} #word counts for each class

        #initializing bag of words for each class
        for name in self.class_count:
            self.word_counts[name] = {}

        #going through each text, building vocabulary and bag of words
        for doc, target in zip(data, targets):
            tokens = self.helper.tokenize_and_clean(doc)

            for word in tokens:
                #add word to global vocabulary
                if(word not in self.vocab):
                    self.vocab.add(word)

                if(word not in self.word_counts[target]):
                    self.word_counts[target][word] = 1
                else:
                    self.word_counts[target][word] += 1

        #number of words in global and class vocabs
        self.vocab_size = len(self.vocab)
        self.class_vocab_size = {}
        for name in self.class_count:
            self.class_vocab_size[name] = len(self.word_counts[name])
          


    def classify(self, data):
        """
        Classify data using trained model.

        Parameters:
            data - string representing a document

        Return:
            predicted class for the input data

        """

        #tokenize data
        tokens = self.helper.tokenize_and_clean(data)

        prob = {}

        #for each class
        for name in self.class_count:

            words = set()

            #initialize with class probability
            prob[name] = self.class_prob[name]

            #get probability for each word
            for word in tokens:
                if(word in self.vocab):
                    if(word not in words):
                        words.add(word)
                        prob[name] += self.get_probability(word, name)

        #class is the one with the highest probability       
        return max(prob, key=lambda key: prob[key])


    def get_probability(self, word, target):
        """
        Get probability from word counts found in the training of model.

        Parameters:
            word: word for which the probability should be calculated
            target: target class for the conditional probability 

        """

        #probability with laplace smoothing
        #(nk+1)/(n+|Vocabulary|)
        #using log

        count = self.word_counts[target].get(word, 0.0) 

        return math.log((count + 1.0)/(self.class_vocab_size[target]+ self.vocab_size))


    def accuracy_test(self, data, target):
        """
        Use the classifier to predict the classes of the given data and 
        return the resulting accuracy

        Parameters:
            data: array containing the data to be used in the test
            target: expected class of the data

        Return
            Accuray for the tests

        """

        corr = 0

        for i in range(0, len(target)):
            if(self.classify(data[i])==target[i]):
                corr+=1

        return (corr/len(target))



        

