from collections import Counter
from textHelper import TextHelper
import math

class NaiveBayesTextClassifier():

    def __init__(self):
        self.helper = TextHelper()

    def fit(self, data, targets):
        self.vocab = set() #global vocabulary
        self.class_count  = Counter(targets) #number of instances of each class
        self.class_prob = {}
        
        for c in self.class_count:
            self.class_prob[c] = float(self.class_count[c])/len(targets) 
            #use log of probability so classifier doesnt underflow
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


        #return class
        #for name in self.class_count:
            #print(name,":",prob[name])
       
        return max(prob, key=lambda key: prob[key])


    def get_probability(self, word, target):

        #probability with laplace smoothing
        #(nk+1)/(n+|Vocabulary|)
        count = self.word_counts[target].get(word, 0.0) 


        return math.log((count + 1.0)/(self.class_vocab_size[target]+ self.vocab_size))


    def accuracy_test(self, data, target):

        corr = 0

        for i in range(0, len(target)):
            if(self.classify(data[i])==target[i]):
                corr+=1

        return (corr/len(target))



        

