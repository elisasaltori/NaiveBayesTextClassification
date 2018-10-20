import os
from collections import Counter
from textHelper import TextHelper
from naiveBayesTextClassifier import NaiveBayesTextClassifier

#read documents by class

#clean documents, tokenize

#test algorithm
#10 fold cross validation
#   fit naive bayes
#       count words and get vocabulary
#       get probabilities
#   stest



TARGET_NAMES = ["CBR", "ILP", "RI"]
DATA_DIR = "docs"

def main():

    data, target = get_documents()

    print(data[10], target[10])

    clf = NaiveBayesTextClassifier()
    
    clf.fit(data, target)

    ac = 0
    for i in range(0,len(target)):
        output = clf.classify(data[i])
        if(output==target[i]):
            ac+=1
            #print("right")
        #print(i,"- expected",target[i],"got", output)
        input()

    print(ac/len(target))
    """
    """
    #print(clean_tokens)



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