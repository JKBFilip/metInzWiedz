import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# dane
data = [
    ['Słoneczna', 'Gorąco', 'Wysoka', 'Słaby', 'Nie'],
    ['Słoneczna', 'Gorąco', 'Wysoka', 'Mocny', 'Nie'],
    ['Pochmurna', 'Gorąco', 'Wysoka', 'Słaby', 'Tak'],
    ['Deszczowa', 'Łagodnie', 'Wysoka', 'Słaby', 'Tak'],
    ['Deszczowa', 'Chłodno', 'Normalna', 'Słaby', 'Tak'],
    ['Deszczowa', 'Chłodno', 'Normalna', 'Mocny', 'Nie'],
    ['Pochmurna', 'Chłodno', 'Normalna', 'Mocny', 'Tak'],
    ['Słoneczna', 'Łagodnie', 'Wysoka', 'Słaby', 'Nie'],
    ['Słoneczna', 'Chłodno', 'Normalna', 'Słaby', 'Tak'],
    ['Deszczowa', 'Łagodnie', 'Normalna', 'Słaby', 'Tak'],
    ['Słoneczna', 'Łagodnie', 'Normalna', 'Mocny', 'Tak'],
    ['Pochmurna', 'Łagodnie', 'Wysoka', 'Mocny', 'Tak'],
    ['Pochmurna', 'Gorąco', 'Normalna', 'Słaby', 'Tak'],
    ['Deszczowa', 'Łagodnie', 'Wysoka', 'Mocny', 'Nie']
]

columns = ['Pogoda', 'Temperatura', 'Wilgotność', 'Wiatr', 'Gram_w_Tenisa']
df = pd.DataFrame(data, columns=columns)

# etykiety
le = preprocessing.LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# dane wejscie i wyjscie
X = df.drop('Gram_w_Tenisa', axis=1)
y = df['Gram_w_Tenisa']

# model drzewa
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf = clf.fit(X, y)

#wizualizacja
plt.figure(figsize=(20,10))
plot_tree(clf,
          feature_names=['Pogoda', 'Temperatura', 'Wilgotność', 'Wiatr'],
          class_names=['Nie', 'Tak'],
          filled=True, rounded=True)
plt.savefig('drzewo_decyzyjne_matplotlib.png')


