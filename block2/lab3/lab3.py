import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def fit(self, X, y):
        """
        Обучение модели на данных
        :param X: Матрица независимых переменных
        :param y: матрица зависимых переменных
        """
        X = np.insert(X, 0, 1, axis=1) # добавим столбец единиц в начало матрицы
        XT_X_inv = np.linalg.inv(X.T @ X)
        weights = np.linalg.multi_dot([XT_X_inv, X.T, y]) # Расчет вектора весов
        self.bias, self.weights = weights[0], weights[1:]

    def predict(self, X_test):
        return X_test @ self.weights + self.bias # формула линейной регрессии

# Z-нормализация
def z_normalization(data):
    norm_data = data.copy()
    data_col = norm_data.columns
    for col in data_col:
        mean = norm_data[col].mean() # среднее значение
        std = norm_data[col].std()   # стандартное отклонение
        norm_data[col] = (norm_data[col] - mean) / std # формула Z-нормализации
    return norm_data

# Коэффициент детерминации (качество модели)
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Загрузка данных
data = pd.read_csv('california_housing_train.csv')

# Заполнение пропусков
data.fillna(data.mean(), inplace=True)

# Статистика
print("Основные статистики по датасету:")
print(data.describe())

# Визуализация
data.hist(figsize=(10, 8))
plt.show()

normalized_data = z_normalization(data) # нормализуем данные

# Разделение данных на обучающий и тестовый наборы данных
np.random.seed(42)
shuffled_indices = np.random.permutation(normalized_data.index) # перемешиваем данные
test_size = int(0.2 * len(normalized_data))  # обрезаем 20% данных для обучения модели
test_indices = shuffled_indices[:test_size]
train_indices = shuffled_indices[test_size:]

test_data = normalized_data.loc[test_indices]
train_data = normalized_data.loc[train_indices]

X_train = train_data.drop('median_house_value', axis=1) # заменим целевой столбец на 1
y_train = train_data['median_house_value']
X_test = test_data.drop('median_house_value', axis=1) # заменим целевой столбец на 1
y_test = test_data['median_house_value']

# Модель 1: Все признаки
linear_regression_all = LinearRegression()
linear_regression_all.fit(X_train, y_train)
y_pred_all = linear_regression_all.predict(X_test)
r2_all = r2_score(y_test, y_pred_all)

# Модель 2: Без "housing_median_age"
X_train_model_2 = X_train.drop('housing_median_age', axis=1) # заменим целевой столбец на 1
X_test_model_2 = X_test.drop('housing_median_age', axis=1) # заменим целевой столбец на 1

linear_regression_model_2 = LinearRegression()
linear_regression_model_2.fit(X_train_model_2, y_train)
y_pred_model_2 = linear_regression_model_2.predict(X_test_model_2)
r2_no_housing_median_age = r2_score(y_test, y_pred_model_2)

# Модель 3: Используем только "total_rooms" и "total_bedrooms"
X_train_model_3 = X_train[['total_rooms', 'total_bedrooms']]
X_test_model_3 = X_test[['total_rooms', 'total_bedrooms']]

linear_regression_model_3 = LinearRegression()
linear_regression_model_3.fit(X_train_model_3, y_train)
y_pred_model_3 = linear_regression_model_3.predict(X_test_model_3)
r2_model_3 = r2_score(y_test, y_pred_model_3)

# Модель 4: Используем только "total_rooms"
X_train_model_4 = X_train[['total_rooms']]
X_test_model_4 = X_test[['total_rooms']]

linear_regression_model_4 = LinearRegression()
linear_regression_model_4.fit(X_train_model_4, y_train)
y_pred_model_4 = linear_regression_model_4.predict(X_test_model_4)
r2_model_4 = r2_score(y_test, y_pred_model_4)

print(f"Модель 1 (Все признаки) - R^2: {r2_all}")
print(f"Модель 2 (Без 'housing_median_age') - R^2: {r2_no_housing_median_age}")
print(f"Модель 3 (total_rooms и total_bedrooms) - R^2: {r2_model_3}")
print(f"Модель 4 (Только total_rooms) - R^2: {r2_model_4}")

# Синтетический признак
X_train_model_5 = X_train.copy()
X_test_model_5 = X_test.copy()

X_train_model_5["rooms_per_household"] = X_train_model_5["total_rooms"] / X_train_model_5["households"]
X_test_model_5["rooms_per_household"] = X_test_model_5["total_rooms"] / X_test_model_5["households"]

linear_regression_model_5 = LinearRegression()
linear_regression_model_5.fit(X_train_model_5, y_train)
y_pred_model_5 = linear_regression_model_5.predict(X_test_model_5)
r2_model_5 = r2_score(y_test, y_pred_model_5)

print(f"Модель 5 (С синтетическим признаком) - R^2: {r2_model_5}")
