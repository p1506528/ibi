#Travail de Thomas Bourg et Théo Buttez

import gzip # pour décompresser les données
import pickle # pour désérialiser les données
import numpy # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import random

  
if __name__ == '__main__':
    # nombre d'image lues à chaque fois dans la base d'apprentissage (laisser à 1 sauf pour la question optionnelle sur les minibatchs)
    TRAIN_BATCH_SIZE = 1
    # on charge les données de la base MNIST
    data = pickle.load(gzip.open('mnist.pkl.gz'),encoding='latin1')
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
    poids_cache = numpy.random.rand(785,10)
    poids_sortie = numpy.random.rand(10,10)
    taux = 0.001
    pas = 500000
    
    print('loaded')
    
    train = []
    for image, label in train_loader:
        image = image[0,:].numpy()
        image = numpy.append(image, 1)
        label = label[0,:].numpy()
        train.append((image, label))

    print('training')
    
    for i in range(pas):
        print(i)
        record = random.choice(train)
        img = record[0]
        label = record[1].reshape((1,10))
        exp = -img.dot(poids_cache)
        activite_cache = 1 / (1 + numpy.exp(exp))
        print(activite_cache.shape)
        activite_sortie = poids_sortie.dot(activite_cache)
        print(activite_sortie.shape)
        print(label.shape)
        erreur_sortie = label - activite_sortie
        print(erreur_sortie.shape)
        erreur_cache = activite_cache * (1 - activite_cache) * erreur_sortie.dot(poids_sortie)
        print(erreur_cache.shape)
        diff_cache = taux * erreur_sortie * activite_sortie
        print(diff_cache.shape) 
        diff = label - activite
        img = img.reshape((785,1))
        rectif = taux * img * diff
        poids += rectif
    
    print('test')
    
    n = 0
    g = 0
    for image, label in test_loader:
        n += 1
        img = image[0,:].numpy()
        img = numpy.append(img, 1)
        activite = img.dot(poids)
        if(numpy.argmax(activite) == numpy.argmax(label[0,:].numpy())):
            g = g + 1
    print(g/n)