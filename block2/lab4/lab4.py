import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib.colors import ListedColormap


file_path = 'WineDataset.csv'
data = pd.read_csv(file_path)

# Заполняем все отсутствующие данные средним значением
data = data.fillna(data.mean())

# Применяем кодирование Label Encoding для столбца 'Wine'
data['Wine'], unique = pd.factorize(data['Wine'])


num_cols = data.select_dtypes(include=['float64']).columns.tolist()  
num_cols = [col for col in num_cols if col != 'Wine'] 


# MinMax нормировка
data[num_cols] = (data[num_cols] - data[num_cols].min()) / (data[num_cols].max() - data[num_cols].min())


data_statistics = data.describe()
data_statistics = data_statistics.rename(index={
    'count': 'Всего',
    'mean': 'Среднее значение',
    'std' : 'Стандартное отклонение',
    'min' : 'Минимальное значение',
    'max' : 'Максимальное значение'
})
print(data_statistics)


fig = plt.figure(figsize=(14, 7))  
ax1 = fig.add_subplot(121, projection='3d') 

x = data['Alcohol']
y = data['Malic Acid']
z = data['Color intensity']

colors = ['#00BFFF',  
          '#00FA9A',  
          '#8A2BE2']  

cmap = ListedColormap(colors) 

scatter1 = ax1.scatter(x, y, z, c=data['Wine'], cmap=cmap, vmin=0, vmax=2) 

ax1.set_xlabel('Алкоголь')
ax1.set_ylabel('Яблочная кислота')
ax1.set_zlabel('Интенсивность цвета')

cbar1 = fig.colorbar(scatter1, ticks=[0, 1, 2], pad = 0.1)  
cbar1.set_ticklabels(['Wine 0', 'Wine 1', 'Wine 2']) 


ax2 = fig.add_subplot(122, projection='3d')

x = data['Ash']
y = data['Alcalinity of ash']
z = data['Magnesium']

colors = ['#00BFFF', 
          '#00FA9A',
          '#8A2BE2'] 
cmap = ListedColormap(colors) 

scatter2 = ax2.scatter(x, y, z, c=data['Wine'], cmap=cmap, vmin=0, vmax=2) 

ax2.set_xlabel('Зола')
ax2.set_ylabel('Щелочность золы')
ax2.set_zlabel('Магний')

cbar2 = fig.colorbar(scatter2, ticks=[0, 1, 2], pad = 0.1) 
cbar2.set_ticklabels(['Wine 0', 'Wine 1', 'Wine 2'])

plt.tight_layout()
plt.show()


features = ['Color intensity', 'Proline', 'Total phenols']
mean_values = data.groupby('Wine')[features].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

width = 0.25 
x = np.arange(len(mean_values)) 

bars1 = ax.bar(x - width, mean_values['Color intensity'], width, label='Интенсивность цвета', color='#FF69B4')
bars2 = ax.bar(x, mean_values['Proline'], width, label='Пролин', color='#8A2BE2')
bars3 = ax.bar(x + width, mean_values['Total phenols'], width, label='Всего фенолов', color='#00BFFF')

ax.set_ylabel('Средние значения')
ax.set_title('Средние значения интенсивности цвета, количества фенолов и пролина по классам вина')
ax.set_xticks(x)
ax.set_xticklabels(['Wine 0', 'Wine 1', 'Wine 2']) 

ax.legend()

plt.tight_layout()
plt.show()



np.random.seed(43)

# Разделение данных на обучающую и тестовую выборку в соотношении 80/20
shuffled_indices = np.random.permutation(data.index)
train_size = int(0.8 * len(data))
test_indices = shuffled_indices[train_size:]
train_indices = shuffled_indices[:train_size]

train_data = data.loc[train_indices]
test_data = data.loc[test_indices]

# Получение обучающей и тестовой выборки
X_train = train_data.drop('Wine', axis=1) 
y_train = train_data['Wine']
X_test = test_data.drop('Wine', axis=1)
y_test = test_data['Wine']


# Расстояние по Евклидовой метрике
def euclidian_distance(row1, row2):
    distance = 0
    for i in range(len(row1)):
        distance += (row1.iloc[i] - row2.iloc[i]) ** 2
    return math.sqrt(distance)

# Метод KNN
def the_KNN_method(x_train, x_test, y_train, k):
    predictions = []
    for i in range (len(x_test)):
        distances = []
        k_distanses = []
        wine_types = []
        for j in range(len(x_train)):
            distances.append((euclidian_distance(x_train.iloc[j], x_test.iloc[i]), y_train.iloc[j]))

        distances.sort(key=lambda x: x[0])
        k_distanses = distances[:k]

        wine_types = [label for _, label in k_distanses]
        
        predictions.append(max(set(wine_types), key=wine_types.count))

    return predictions


def accuracy(test, test_prediction):
    correct = 0
    for i in range(len(test)):
        if test.iloc[i] == test_prediction[i]:
            correct += 1
    return (correct / len(test))


predictions1 =  the_KNN_method(X_train, X_test, y_train, 5)
print("Модель, обученная на всех признаках. k = 5")
print("Точность: " + str(accuracy(y_test, predictions1)))
print("Матрица ошибок")
crosstab = pd.crosstab(y_test, predictions1)
print(crosstab)
        

shuffled_indices = np.random.permutation(data.index)
train_size = int(0.8 * len(data))
test_indices = shuffled_indices[train_size:]
train_indices = shuffled_indices[:train_size]

train_data = data.loc[train_indices]
test_data = data.loc[test_indices]

# Получение обучающей и тестовой выборки
X_train = train_data[['Alcohol', 'Color intensity']]
y_train = train_data['Wine']
X_test = test_data[['Alcohol', 'Color intensity']]
y_test = test_data['Wine']


print()
print("Модель 1, построенная на признаках Alcohol и Color intensity")
print("k = 3")
predictions2 =  the_KNN_method(X_train, X_test, y_train, 3)
print("Точность: " + str(accuracy(y_test, predictions2)))
print("Матрица ошибок")
crosstab = pd.crosstab(y_test, predictions2)
print(crosstab)

print()
print("k = 5")
predictions3 =  the_KNN_method(X_train, X_test, y_train, 5)
print("Точность: " + str(accuracy(y_test, predictions3)))
print("Матрица ошибок")
crosstab = pd.crosstab(y_test, predictions3)
print(crosstab)


print()
print("k = 7")
predictions4 =  the_KNN_method(X_train, X_test, y_train, 7)
print("Точность: " + str(accuracy(y_test, predictions4)))
print("Матрица ошибок")
crosstab = pd.crosstab(y_test, predictions4)
print(crosstab)



shuffled_indices = np.random.permutation(data.index)
train_size = int(0.8 * len(data))
train_indices = shuffled_indices[:train_size]
test_indices = shuffled_indices[train_size:]

train_data = data.loc[train_indices]
test_data = data.loc[test_indices]

feature_columns = data.columns.drop('Wine') 
selected_features = np.random.choice(feature_columns, size=3, replace=False)

X_train = train_data[selected_features]
y_train = train_data['Wine']
X_test = test_data[selected_features]
y_test = test_data['Wine']


print()
print("Модель 2, построенная на признаках " + str(selected_features))
print("k = 3")
predictions1 =  the_KNN_method(X_train, X_test, y_train, 3)
print("Точность: " + str(accuracy(y_test, predictions1)))
print("Матрица ошибок")
crosstab = pd.crosstab(y_test, predictions1)
print(crosstab)

print()
print("k = 5")
predictions2 =  the_KNN_method(X_train, X_test, y_train, 5)
print("Точность: " + str(accuracy(y_test, predictions2)))
print("Матрица ошибок")
crosstab = pd.crosstab(y_test, predictions2)
print(crosstab)

print()
print("k = 7")
predictions3 =  the_KNN_method(X_train, X_test, y_train, 7)
print("Точность: " + str(accuracy(y_test, predictions3)))
print("Матрица ошибок")
crosstab = pd.crosstab(y_test, predictions3)
print(crosstab)