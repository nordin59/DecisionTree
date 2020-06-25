import numpy as np
import random

def load_iris_dataset(train_ratio):

    random.seed(1) 
    conversion_labels = {'Iris-setosa': 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}

    #On importe nos données que l'on met dans une liste. Chaque élément de la liste est une liste des valeurs de chaue obs.
    f = open('datasets/bezdekIris.data', 'r')
    database=f.readlines()
    Liste_d=[]
    for x in database:
        obs=x.split(sep=",")
        Liste_d.append(obs)

    #On convertit les string en float et les classe en 0,1,2
    for j in range(len(Liste_d)-1):
        if (Liste_d[j][4]=='Iris-setosa\n'):
            Liste_d[j][4]=0
        elif (Liste_d[j][4]=='Iris-versicolor\n'):
            Liste_d[j][4]=1
        elif (Liste_d[j][4]=='Iris-virginica\n'):
            Liste_d[j][4]=2
        else:
            Liste_d[j][4]="NULLL"
        for k in range(4):
            Liste_d[j][k]=float(Liste_d[j][k])

    #On supprime le dernier element qui correspond à\n
    del Liste_d[len(Liste_d)-1]

    #On procède au mélange de notre dataset
    random.shuffle(Liste_d)
    #On convertit notre ratio
    ratio=train_ratio/100

    #On découpe notre ensemble en train et test
    train_set=Liste_d[0:int(len(Liste_d)*ratio)]
    test_set=Liste_d[int(len(Liste_d)*ratio):len(Liste_d)]

    #On separe les labels de nos données
    train=train_set
    train_labels=[]
    for k in range(len(train)):
        train_labels.append(train[k][4])
        del train[k][4]

    test=test_set
    test_labels=[]
    for k in range(len(test)):
        test_labels.append(test[k][4])
        del test[k][4]
 
    return (train, train_labels, test, test_labels)
	
	
