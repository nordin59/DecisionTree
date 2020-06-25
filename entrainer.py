import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import time
import load_datasets
import classifieur  # importer la classe de l'Arbre de Décision

train_ratio = 70

train_iris, train_labels_iris, test_iris, test_labels_iris = load_datasets.load_iris_dataset(train_ratio)

def conversion(train , train_labels ,  test , test_labels):
    train  = pd.DataFrame(train)
    train["labels"] = train_labels
    features_train = list(train)
    features_train.pop()
    test  = pd.DataFrame(test)
    test["labels"] = test_labels
    features_test = list(train)
    features_test.pop()
    return(train, test , features_train , features_test)

train_iris,test_iris,features_train_iris,feature_test_iris= conversion(train_iris, train_labels_iris, test_iris, test_labels_iris)

print("\n\n-------DATASET IRIS---------\n\n")

#On crée nos modèles
model_tree = classifieur.DecisionTree(30)


#On crée notre arbre
t0 = time.time()
arbre_iris = model_tree.arbre( train_iris , train_iris , features_train_iris , labels = "labels")
t1 = time.time()
print("temps d'entrainement",t1-t0)

#On fais la prediction sur l'entrainement
training_iris = model_tree.train( train_iris , arbre_iris )
conf_train_iris = model_tree.confusion_matrix(train_iris , arbre_iris)



#On test notre modèle
testing_iris = model_tree.test( test_iris , arbre_iris)
conf_train_iris = model_tree.confusion_matrix(test_iris , arbre_iris)

#Sur un seul exemple
t0 = time.time()
exemple = test_iris.iloc[1,:]
predict_exemple = model_tree.predict( exemple, arbre_iris, default=1)
print("prediction",predict_exemple)
t1 = time.time()

print("temps de prediction d'un exemple",t1-t0)

print("\n\n-------Courbe d'apprentissage---------\n\n")

def courbe_apprentissage( train , test , features , profondeur , coupe, labels = "labels"):
    n = len(train)
    m = int(n/coupe)
    Accuracy = []
    indice = []
    for i in range(2,m+1):
        indice.append(i*coupe)
        sous_train = train.iloc[0:(i+1)*coupe,:]
        model_tree = classifieur.DecisionTree(profondeur)
        arbre = model_tree.arbre( sous_train , sous_train , features , labels = "labels")
        tester = model_tree.test2( test , arbre)
        Accuracy.append(tester)
    plt.plot(indice, Accuracy)
    plt.ylabel('accuracy on dataset')
    plt.xlabel('size dataset')
    plt.show()



profondeur = 30
plot_iris = courbe_apprentissage( train_iris , test_iris , features_train_iris , profondeur , 2 , labels = "labels")
