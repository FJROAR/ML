# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 23:37:44 2019

@author: fjroa
"""

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

#Visualización de dígitos escritos a mano

flg, axes = plt.subplots(2, 5, figsize = (10, 5),
                         subplot_kw = {'xticks':(), 'yticks':()})

for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)

#Como se observa, las cifras son distintas
plt.gray() 
plt.matshow(digits.images[0]) 
plt.show()

plt.gray() 
plt.matshow(digits.images[20]) 
plt.show()


#Se hace uso de un PCA para representar los datos

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca.fit(digits.data)
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#785188", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120", "#535DBE"]

plt.figure(figsize =(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())          
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())

for i in range(len(digits.data)):
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict = {'weight': 'bold', 'size': 9})
    
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
          

#Se hace uso de un t-SNE para representar los datos

from sklearn.manifold import TSNE

tsne = TSNE(random_state = 42)
digits_tsne = tsne.fit_transform(digits.data)
colors = ["#476A2A", "#785188", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120", "#535DBE"]

plt.figure(figsize =(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())          
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())

for i in range(len(digits.data)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict = {'weight': 'bold', 'size': 9})
    
plt.xlabel("t-SNE feature 0")
plt.ylabel("t-SNE feature 1")
          
