#Travail de Thomas Bourg et Théo Buttez

import gzip # pour décompresser les données
import pickle # pour désérialiser les données
import numpy as np# pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import random
import math
import statistics

  
if __name__ == '__main__':
    # nombre d'image lues à chaque fois dans la base d'apprentissage (laisser à 1 sauf pour la question optionnelle sur les minibatchs)
    TRAIN_BATCH_SIZE = 1
    # on charge les données de la base MNIST
    data = pickle.load(gzip.open('mnist_light.pkl.gz'),encoding='latin1')
    # images de la base d'apprentissage
    train_data = torch.Tensor(data[0][0])
    # labels de la base d'apprentissage
    train_data_label = torch.Tensor(data[0][1])
    # images de la base de test
    test_data = torch.Tensor(data[1][0])
    # labels de la base de test
    test_data_label = torch.Tensor(data[1][1])
    train_dataset = torch.utils.data.TensorDataset(train_data,train_data_label)
    test_dataset = torch.utils.data.TensorDataset(test_data,test_data_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print('loaded')
    taux = 0.001
    pas = 100
    nb_input = 784
    nb_cache = 128                              
    nb_output = 10
    pds_cache = np.zeros((nb_cache, nb_input)) 
    pds_sortie = np.random.randn(nb_output, nb_cache) 
    biais_cache = np.zeros((nb_cache))  
    biais_sortie = np.random.randn(nb_output)   

    def sigmoid(y):
        return 1. / (1. + np.exp(-y))

    def der_sigmoid(y):
        return np.multiply(y, (1 - y))
    
    train = []
    for image, label in train_loader:
        image = image[0,:].numpy()
        label = label[0,:].numpy()
        train.append((image, label))

    print('training')
    
    for i in range(pas):
        print(i)
        if i < len(train):
            record = train[i]
        else:
            record = random.choice(train)
        img = record[0]
        label = record[1]
        
        activite_cache = sigmoid(np.dot(pds_cache, img) + biais_cache)
        activite_sortie = np.dot(pds_sortie, activite_cache) + biais_sortie
        
        erreur_sortie = label - activite_sortie
        erreur_cache = der_sigmoid(activite_cache) * np.dot(erreur_sortie, pds_sortie)
        
        diff_sortie = taux * np.transpose([activite_cache] * nb_output) * erreur_sortie
        pds_sortie += diff_sortie.T
        biais_sortie += taux * erreur_sortie
        diff_cache = taux * np.transpose([img] * nb_cache) * erreur_cache
        pds_cache += diff_cache.T
        biais_cache += taux * erreur_cache

    print('test')

    def test():
        accuracy = 0
        for img, label in test_loader:
            y_h = sigmoid(np.dot(pds_cache, img[0,:].numpy()) + biais_cache)    
            y_o = np.dot(pds_sortie, y_h) + biais_sortie
            accuracy += np.argmax(y_o) == np.argmax(label[0,:].numpy())
        return accuracy / len(test_data)

    print(test())