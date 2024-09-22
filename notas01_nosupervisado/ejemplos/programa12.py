# -*- coding: utf-8 -*-
"""
Autor fjra

"""


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

#https://stackoverflow.com/questions/57242208/how-to-resolve-the-error-module-umap-has-no-attribute-umap-i-tried-installi
#pip uninstall umap
#pip install umap-learn

import umap.umap_ as umap

reducer = umap.UMAP(random_state=42)
digits_umap = reducer.fit_transform(digits.data)

colors = ["#476A2A", "#785188", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120", "#535DBE"]

plt.figure(figsize =(10, 10))
plt.xlim(digits_umap[:, 0].min(), digits_umap[:, 0].max())          
plt.ylim(digits_umap[:, 1].min(), digits_umap[:, 1].max())

for i in range(len(digits.data)):
    plt.text(digits_umap[i, 0], digits_umap[i, 1], str(digits.target[i]),
             color = colors[digits.target[i]],
             fontdict = {'weight': 'bold', 'size': 9})
    
plt.xlabel("UMAP feature 0")
plt.ylabel("UMAP feature 1")
