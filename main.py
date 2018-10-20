"""
SCC231 - Introdução a Sistemas Inteligentes
Professor: Alneu de Andrade Lopes
PAE: Fabiana Góes


Projeto - Classificação textual com Naive Bayes

Aluno: Elisa Saltori Trujillo (8551100)


"""

import os
from collections import Counter
from textHelper import TextHelper
from naiveBayesTextClassifier import NaiveBayesTextClassifier
from sklearn.model_selection import KFold
import numpy as np


DATA_DIR = "docs" #folder where the corpus is

def main():
    """
    Train and test naive bayes classifier on CBR-IL-PIR corpus.
    
    Read documents from DATA_DIR directory.
    Run 10-fold cross validation test with naive bayes.
    Print results of test

    """
    print("Reading documents...")
    data, target = get_documents()
    print("Done!")

    accuracies = test_naiveBayes(data, target)

    print()
    print("Results:")
    print(accuracies)
    print("Mean accuracy:", sum(accuracies) / len(accuracies))
    print("Variance:", np.std(accuracies))



def test_naiveBayes(data, target):
    """
    Apply 10-fold cross validation to Naive Bayes algorithm
    
    Parameters:
        data - array with the text of each document
        target - array with the class attribute of each document

    Return:
        Array with the accuracy of each test

    """
    print("Beginning naive Bayes test")

    #divide data into 10 folds
    kfold = KFold(10, True)

    accuracies = [] #accuracy results

    #conversion to arrays to be able to index it with train, test lists
    data_array = np.asarray(data)
    target_array = np.asarray(target)

    
    i=1 #counter for folds

    #for each fold
    for train, test in kfold.split(data):
        print("...working on fold", i)
        i+=1
        clf = NaiveBayesTextClassifier()
        #train classifier
        clf.fit(data_array[train],target_array[train])
        #test accuracy with test fold
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



if __name__ == "__main__":
    main()