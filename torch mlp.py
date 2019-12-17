#Travail de Thomas Bourg et Théo Buttez

import gzip # pour décompresser les données
import pickle # pour désérialiser les données
import numpy as np# pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data
import random
import math
import statistics
from sklearn.metrics import classification_report, confusion_matrix


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def compute_loss(Y, Y_hat):

    m = Y.shape[1]
    L = -(1./m) * ( np.sum( np.multiply(np.log(Y_hat),Y) ) + np.sum( np.multiply(np.log(1-Y_hat),(1-Y)) ) )

    return L

  
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
    W1 = np.random.randn(100,28*28)
    b1 = np.zeros((100, 1))
    W2 = np.random.randn(1, 100)
    b2 = np.zeros((1, 1))
    learning_rate = 0.1
    m = 28*28
    pas = 100
    
    print('loaded')
    
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
        label = record[1].reshape((1,10))
        Z1 = np.matmul(W1, img) + b1
        A1 = sigmoid(Z1)
        Z2 = np.matmul(W2, A1) + b2
        A2 = sigmoid(Z2)

#        cost = compute_loss(label, A2)

        dZ2 = A2-label
        dW2 = (1./m) * np.matmul(dZ2, A1.T)
        db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(W2.T, dZ2)
        dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
        dW1 = (1./m) * np.matmul(dZ1, img.T)
        db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1

        if i % 100 == 0:
            print("Epoch", i, "cost: ", cost)

    print("Final cost:", cost)
    
    print('test')
    
    Z1 = np.matmul(W1, X_test) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)

    predictions = (A2>.5)[0,:]
    labels = (y_test == 1)[0,:]

    print(confusion_matrix(predictions, labels))
    print(classification_report(predictions, labels))

    
