import os
from collections import Counter
from textHelper import TextHelper
from naiveBayesTextClassifier import NaiveBayesTextClassifier
from sklearn.model_selection import KFold
import numpy as np

#read documents by class

#clean documents, tokenize

#test algorithm
#10 fold cross validation
#   fit naive bayes
#       count words and get vocabulary
#       get probabilities
#   stest


DATA_DIR = "docs"

def main():

    print("Reading documents...")
    data, target = get_documents()
    print("Done!")

    clf = NaiveBayesTextClassifier()
    
    clf.fit(data, target)

    accuracies = test_naiveBayes(data, target)

    print()
    print("Results:")
    print(accuracies)
    print("Mean accuracy:", sum(accuracies) / len(accuracies))
    print("Variance:", np.std(accuracies))



def test_naiveBayes(data, target):

    print("Beginning naive Bayes test")

    #divide data into 10 folds
    kfold = KFold(10, True)
    accuracies = []

    data_array = np.asarray(data)
    target_array = np.asarray(target)

    
    i=1

    #for each fold
    for train, test in kfold.split(data):
        print("...working on fold", i)
        i+=1
        clf = NaiveBayesTextClassifier()
        clf.fit(data_array[train],target_array[train])
        accuracies.append(clf.accuracy_test(data_array[test], target_array[test]))

    print("Done!")

    return accuracies



def get_documents():
    """
    Read documents from data_dir directory and mark their 
    """

    data = []
    target = []

    #get documents in data folder
    for filename in os.listdir(DATA_DIR):


        #read document and add it to array
        with open(os.path.join(DATA_DIR, filename), encoding="latin-1") as f:
                data.append(f.read())

                #class of the document
                file_class = filename.split('-')[0]
                target.append(file_class)

    return data, target



main()