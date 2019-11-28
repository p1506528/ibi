#Travail de Thomas Bourg et Théo Buttez

import gzip # pour décompresser les données
import pickle # pour désérialiser les données
import numpy # pour pouvoir utiliser des matrices
import matplotlib.pyplot as plt # pour l'affichage
import torch,torch.utils.data


# fonction qui va afficher l'image située à l'index index
def affichage(image,label):
    # on récupère à quel chiffre cela correspond (position du 1 dans label)
    label = numpy.argmax(label)
    # on crée une figure
    plt.figure()
    # affichage du chiffre
    # le paramètre interpolation='nearest' force python à afficher chaque valeur de la matrice sans l'interpoler avec ses voisines
    # le paramètre cmap définit l'échelle de couleur utilisée (ici noire et blanc)
    plt.imshow(image.reshape((28,28)),interpolation='nearest',cmap='binary')
    # on met un titre
    plt.title('chiffre '+str(label))
    # on affichage les figures créées
    plt.show()

  
# c'est ce qui sera lancé lors que l'on fait python lecture_data_3.py
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
    # on crée la base de données d'apprentissage (pour torch)
    train_dataset = torch.utils.data.TensorDataset(train_data,train_data_label)
    # on crée la base de données de test (pour torch)
    test_dataset = torch.utils.data.TensorDataset(test_data,test_data_label)
    # on crée le lecteur de la base de données d'apprentissage (pour torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    # on crée le lecteur de la base de données de test (pour torch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # 10 fois
    """
    for i in range(0,10):
        # on demande les prochaines données de la base
        (_,(image,label)) = enumerate(train_loader).__next__()
        # on les affiche
        affichage(image[0,:].numpy(),label[0,:].numpy())
        """
    # NB pour lire (plus proprement) toute la base (ce que vous devrez faire dans le TP) plutôt utiliser la formulation suivante
#    for image,label in train_loader:
#        affichage(image[0,:].numpy(),label[0,:].numpy())

    #On initialise les poids aléatoirement
    poids = numpy.random.rand(785,10)
    taux = 0.01
    
    #print(poids)

    #On prend les images une par une dans la base d'apprentissage
    for image, label in train_loader:
        img = image[0,:].numpy()
        img = numpy.append(img, 1)
        activite = img.dot(poids)
        print(activite)
        diff = label[0,:].numpy() - activite
        diff = diff.reshape((1,10))
        img = img.reshape((785,1))
        rectif = taux * img * diff
        #print(rectif.shape)
        #print("coucou")
        poids -= rectif
    
    #print(poids)
    
    (_,(image,label)) = enumerate(test_loader).__next__()
    img = image[0,:].numpy()
    img = numpy.append(img, 1)
    activite = img.dot(poids)
    print(activite)
    print(label[0,:].numpy())
