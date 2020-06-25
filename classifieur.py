
import numpy as np
import pandas as pd


# le nom de votre classe
# DecisionTree le modèle des arbres de decision

class DecisionTree: #nom de la class à changer

    def __init__(self, profondeur):
    	self.profondeur = profondeur

    def Gain(self , train , attribut , labels = "labels" ):
        total_entropy = self.entropy(train[labels])
        vals , counts = np.unique( train[attribut] , return_counts=True )
        Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*self.entropy(train.where(train[attribut]==vals[i]).dropna()[labels]) for i in range(len(vals))])
        gain = total_entropy - Weighted_Entropy
        return gain
	
	#Fonction qui calcule l'entropie pour un attribut, ici une colonne
    def entropy (self , target_col):
        elements,counts = np.unique(target_col,return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy


    def arbre(self , train , original_train , features , labels = "labels" , parent_node_class = None , prof = 0):
        if len(np.unique(train[labels])) <= 1:
            return np.unique(train[labels])[0]
        elif len(train)==0:
            return np.unique(original_train[labels])[np.argmax(np.unique(original_train[labels],return_counts=True)[1])]
        elif len(features) == 0:
            return parent_node_class
        else:
            while (prof<self.profondeur):
                parent_node_class = np.unique(train[labels])[np.argmax(np.unique(train[labels],return_counts=True)[1])]
                item_values = [self.Gain(train , feature , labels ) for feature in features] #Return the information gain values for the features in the dataset
                best_feature_index = np.argmax(item_values)
                best_feature = features[best_feature_index]
                tree = {best_feature:{}}
                features = [i for i in features if i != best_feature]
                for value in np.unique(train[best_feature]):
                    val = value
                    sub_data = train.where(train[best_feature] == val).dropna()
                    subtree = self.arbre(sub_data , original_train , features, labels , parent_node_class , prof+1)
                    tree[best_feature][val] = subtree
                return(tree) 



    def predict( self , input , tree, default = 1):
        for key in list(input.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][input[key]] 
                except:
                    return default
                result = tree[key][input[key]]
                if isinstance(result,dict):
                    return self.predict( input , result)
                else:
                    return result

    def train( self , train , tree ):
        queries = train.iloc[:,:-1].to_dict(orient = "records")
        predicted = pd.DataFrame(columns=["predicted"]) 
        for i in range(len(train)):
            predicted.loc[i,"predicted"] = self.predict(queries[i],tree,1.0)
        print('The train prediction accuracy is: ',(np.sum(predicted["predicted"] == train["labels"])/len(train))*100,'%')

    def test( self,  test , tree ):
        queries = test.iloc[:,:-1].to_dict(orient = "records")
        predicted = pd.DataFrame(columns=["predicted"]) 
        for i in range(len(test)):
            predicted.loc[i,"predicted"] = self.predict(queries[i],tree,1.0)
        print('The test prediction accuracy is: ',(np.sum(predicted["predicted"] == test["labels"])/len(test))*100,'%')
	
    def test2( self,  test , tree ):
        queries = test.iloc[:,:-1].to_dict(orient = "records")
        predicted = pd.DataFrame(columns=["predicted"]) 
        for i in range(len(test)):
            predicted.loc[i,"predicted"] = self.predict(queries[i],tree,1.0)
        accurate = (np.sum(predicted["predicted"] == test["labels"])/len(test))*100
        return(accurate)

    def confusion_matrix( self,  data , tree ):
        queries = data.iloc[:,:-1].to_dict(orient = "records")
        predicted = pd.DataFrame(columns=["predicted"]) 
        mod = len(np.unique(data["labels"]))
        confusion_matrix=np.zeros( (mod, mod) )
        for i in range(len(data)):
            predicted.loc[i,"predicted"] = self.predict(queries[i],tree,1.0)
            a = int(predicted.loc[i,"predicted"])
            b = int(data.loc[i,"labels"])
            if (a == b):
                confusion_matrix[a][b]+=1
            else:
                confusion_matrix[a][b]+=1
        print('Matrice de confusion: ',confusion_matrix)
