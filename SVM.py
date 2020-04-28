# packagers for data analysis

import numpy as np
import pandas as pd
from sklearn import svm

# packagers for data visualising

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.2)

recipes = pd.read_csv('c vs mf.csv')
print(recipes.head())
print("....................................................")

# plot our data

# sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 20})
# plt.show()

# format of pre process of our data

type_label = np.where(recipes['Type'] == 'Muffin', 0, 1)
recipes_features = recipes.columns.values[1:].tolist()
print(recipes_features)
print("....................................................")
ingredients = recipes[['Flour', 'Sugar']].values
print(ingredients)
print("....................................................")

# fit modal

modal = svm.SVC(kernel='linear')
print(modal.fit(ingredients, type_label))
print("....................................................")

# get the separating hyperplane

w = modal.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (modal.intercept_[0]) / w[1]
print(yy)
print("....................................................")

# plot the parallel to the separating hyperplane that pass though the support vector

b = modal.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = modal.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


# sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70});
# plt.plot(xx, yy, linewidth=2, color='black')
# plt.plot(xx, yy_down, 'k--')
# plt.plot(xx, yy_up, 'k--')
# plt.show()


# create function to predict muffin of cupcake

def muffin_or_cupcake(flour, sugar):
    if (modal.predict([[flour, sugar]])) == 0:
        print("Muffin")
    else:
        print("Cupcake")


muffin_or_cupcake(10, 55)

# plotting Predicted data

sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70});
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(10, 55, 'yo', markersize=9)
plt.show()
