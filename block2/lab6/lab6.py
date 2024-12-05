import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

def z_normalization(data, required_value):
    '''
    Z-нормализация
    '''
    norm_data = data[:]
    data_col = norm_data.columns
    data_col = data_col.drop(required_value)
    for col in data_col:
        mean = norm_data[col].mean() # среднее значение
        std = norm_data[col].std()   # стандартное отклонение
        norm_data[col] = (norm_data[col] - mean) / std # формула Z-нормализации
    return norm_data

'''
Чтение данных из файла
'''
file_path = 'diabetes.csv'
required_value = 'Outcome'
data = pd.read_csv(file_path)


'''
заменяем "?" на моду
'''
for column in data.columns:
    mode_value = data[column].mode()[0] 
    data[column] = data[column].replace('?', mode_value)


print()
print(data[data.columns].describe())
print()

'''
Разделение данных на обучающую и тестовую выборку в соотношении 80/20
'''
np.random.seed(19)
data = z_normalization(data, required_value)
train_size = int(0.8 * len(data))
shuffled_indices = np.random.permutation(data.index)
test_indices = shuffled_indices[train_size:]
train_indices = shuffled_indices[:train_size]

train_data = data.loc[train_indices]
test_data = data.loc[test_indices]

'''
Получение обучающей и тестовой выборки
'''
X_train = train_data.drop(required_value, axis=1)
y_train = train_data[required_value]
X_test = test_data.drop(required_value, axis=1)
y_test = test_data[required_value]

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)



class Model :
    def __init__(self, X_train, y_train, X_test, y_test) :
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.log_less = None
        self.koeffs = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.learning_rate = None
        self.iteration_count = None


    def __calculate_predicted_probability__(self, koeffs, features) :
        '''
        Sigmoid function
        Вычисление предсказанных вероятностей
        features -- массив со значениями признаков
        koefs --массив коэффициентов
        '''
        power = koeffs[0]
        for i in range(len(features)) :
            power += koeffs[i + 1] * features[i]
        power = -power

        return 1 / (1 + np.exp(power))


    def __calculate_log_loss__(self, p_array, y_array) :
        '''
        Вычисление функции потери
        p_array -- массив предсказанных вероятностей
        r_array -- массив результатов
        '''
        return -sum(y_array * np.log(p_array) + (1 - y_array) * np.log(1 - np.array(p_array))) / len(y_array)


    def __calculate_gradient__(self, p_array, y_array, feature_array = None) :
        '''
        Вычисление градиента
        p_array -- массив предсказанных вероятностей
        r_array -- массив результатов
        feature_array -- массив значений признаков
        '''
        if feature_array is None :
            return sum(p_array - y_array)
        return sum((p_array - y_array) * feature_array) / len(y_array)


    def __calculate_hessian__(self, p_array, feature_j=None, feature_k=None):
        if feature_j is None:
            feature_j = np.ones(len(p_array))  # Смещение (bias term)
        if feature_k is None:
            feature_k = np.ones(len(p_array))  # Смещение (bias term)

        hessian = 0
        for i in range(len(p_array)):
            p = p_array[i]
            x_j = feature_j.iloc[i] if isinstance(feature_j, pd.Series) else feature_j[i]
            x_k = feature_k.iloc[i] if isinstance(feature_k, pd.Series) else feature_k[i]
            hessian += p * (1 - p) * x_j * x_k

        return hessian


    def train_model(self, learning_rate = 0.01, iteration_count = 100, isNewtonOptimisation = False) :
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        koeffs = [0 for _ in range(self.X_train.shape[1] + 1)]

        for i in range(iteration_count) :
            '''
            Вычисление предсказанных вероятностей (sigmoid function)
            '''
            if i % 100 == 0 : print('.', end='')
            p_array = []
            for j in range(len(self.y_train)) :
                row_series = self.X_train.loc[j]
                row_list = row_series.tolist()
                p_array.append(self.__calculate_predicted_probability__(koeffs, row_list))

            '''
            Вычисление функции потери
            '''
            log_less = self.__calculate_log_loss__(p_array, self.y_train)

            '''
            Вычисление градиентов
            '''
            gradients = []
            gradients.append(self.__calculate_gradient__(p_array, self.y_train))
            for column_name in self.X_train.columns:
                gradients.append(self.__calculate_gradient__(p_array, self.y_train, self.X_train[column_name]))

            '''
            Обновление коэффициентов
            '''
            if i == iteration_count - 1 :
                print('test')
                break
            new_koeffs = []
            if not isNewtonOptimisation :
                '''
                Выполнить градиентный спуск (по умолчанию)
                '''
                for i in range(len(koeffs)) :
                    new_koeffs.append(koeffs[i] - learning_rate * gradients[i])
            else :
                '''
                Выполнить оптимизацию Ньютона
                '''
                hessians = [[0 for _ in range(len(koeffs))] for _ in range(len(koeffs))]

                for j in range(len(koeffs)):
                    for k in range(len(koeffs)):
                        hessians[j][k] = self.__calculate_hessian__(
                            p_array,
                            self.X_train.iloc[:, j - 1] if j > 0 else None,  # Смещение (bias term)
                            self.X_train.iloc[:, k - 1] if k > 0 else None   # Смещение (bias term)
                        )

                # Инвертирование матрицы Гессиана
                hessian_matrix = np.array(hessians)
                hessian_inverse = np.linalg.inv(hessian_matrix)

                # Вычисление шага Ньютона
                gradient_vector = np.array(gradients)
                step = np.dot(hessian_inverse, gradient_vector)

                # Обновление коэффициентов
                new_koeffs = [koeffs[i] - step[i] for i in range(len(koeffs))]

            koeffs = new_koeffs
        
        self.log_less = log_less
        self.koeffs = koeffs

    def sigmoid(self, z):
        # Предотвращение переполнения
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def newton_optimization(self, koeffs, X_train, y_name, gradients):
        # Количество примеров
        N = len(y_name)

        # Количество признаков
        n = len(koeffs) - 1

        # Добавляем единичную компоненту к X_train для учета смещения w_0
        X_train_with_bias = np.column_stack((np.ones(N), X_train))

        # Вычисление предсказанных вероятностей
        z = X_train_with_bias @ koeffs
        y_pred = self.sigmoid(z)

        # Вычисление Гессиана
        H = np.zeros((n + 1, n + 1))
        for i in range(N):
            x_i = X_train_with_bias[i]
            y_i = y_pred[i]
            H += y_i * (1 - y_i) * np.outer(x_i, x_i)
        H /= N

        # Регуляризация для предотвращения вырожденной матрицы
        H += 1e-5 * np.eye(n + 1)

        # Вычисление обратной матрицы Гессиана
        H_inv = np.linalg.inv(H)

        # Обновление коэффициентов
        koeffs_new = koeffs - H_inv @ gradients

        return koeffs_new


    def train_model_new(self, learning_rate = 0.01, iterations_count = [100, 1000, 5000], isNewtonOptimisation = False) :
        self.learning_rate = learning_rate
        self.iteration_count = iterations_count[-1]
        koeffs = [0 for _ in range(self.X_train.shape[1] + 1)]

        koeffs_array = []
        koeffs_index = 0
        log_less_array = []
        for i in range(iterations_count[-1]) :
            '''
            Вычисление предсказанных вероятностей (sigmoid function)
            '''
            if (i + 1) % 500 == 0 :
                print('.')
            # print(i)
            p_array = []
            for j in range(len(self.y_train)) :
                row_series = self.X_train.loc[j]
                row_list = row_series.tolist()
                p_array.append(self.__calculate_predicted_probability__(koeffs, row_list))

            '''
            Вычисление функции потери
            '''
            log_less = self.__calculate_log_loss__(p_array, self.y_train)

            '''
            Вычисление градиентов
            '''
            gradients = []
            gradients.append(self.__calculate_gradient__(p_array, self.y_train))
            for column_name in self.X_train.columns:
                gradients.append(self.__calculate_gradient__(p_array, self.y_train, self.X_train[column_name]))

            '''
            условие выхода и заполнения `koeffs_array`
            '''
            if i + 1 == iterations_count[koeffs_index] :
                koeffs_index += 1
                result_koeffs = koeffs[:]
                koeffs_array.append(result_koeffs)
                # log_less_array.append(log_less)
            if i == iterations_count[-1] - 1 :
                break

            '''
            Обновление коэффициентов
            '''
            new_koeffs = []
            if not isNewtonOptimisation :
                '''
                Выполнить градиентный спуск (по умолчанию)
                '''
                for i in range(len(koeffs)) :
                    new_koeffs.append(koeffs[i] - learning_rate * gradients[i])
            else :
                '''
                Выполнить оптимизацию Ньютона
                '''
                hessians = [[0 for _ in range(len(koeffs))] for _ in range(len(koeffs))]

                for j in range(len(koeffs)):
                    for k in range(len(koeffs)):
                        hessians[j][k] = self.__calculate_hessian__(
                            p_array,
                            self.X_train.iloc[:, j - 1] if j > 0 else None,  # Смещение (bias term)
                            self.X_train.iloc[:, k - 1] if k > 0 else None   # Смещение (bias term)
                        )

                # Инвертирование матрицы Гессиана
                hessian_matrix = np.array(hessians)
                hessian_inverse = np.linalg.inv(hessian_matrix)

                # Вычисление шага Ньютона
                gradient_vector = np.array(gradients)
                step = np.dot(hessian_inverse, gradient_vector)

                # Обновление коэффициентов
                new_koeffs = [koeffs[i] - step[i] for i in range(len(koeffs))]

            koeffs = new_koeffs
        
        self.log_less = log_less
        self.koeffs = koeffs
        print()
        return koeffs_array
    # , log_less_array


    def testing_model(self, threshold = 0.5) :
        '''
        Тестирование на основе тестовых данных.
        threshold -- порог, при котором ответ считается правильным
        '''
        tp = fp = tn = fn = 0
        for i in range(len(self.y_test)) :
            row_series = self.X_test.loc[i]
            row_list = row_series.tolist()
            p = self.__calculate_predicted_probability__(self.koeffs, row_list)
            if p >= threshold :
                if self.y_test[i] == 1 :
                    tp += 1
                else :
                    fp += 1
            else :
                if self.y_test[i] == 0 :
                    tn += 1
                else :
                    fn += 1

        self.accuracy = (tp + tn) / len(y_test)
        self.precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        self.recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) != 0 else 0



model = Model(X_train, y_train, X_test, y_test)


results = []
iterations_count=[100, 500, 1000]
koeffs_array = model.train_model_new(learning_rate=0.01, iterations_count=iterations_count, isNewtonOptimisation=False)
for i in range(len(koeffs_array)) :
    model.koeffs = koeffs_array[i]
    model.testing_model()
    results.append([
        'Градиентный спуск',
        0.01,
        iterations_count[i],
        round(model.log_less, 3),
        round(model.accuracy, 3),
        round(model.precision, 3),
        round(model.recall, 3),
        round(model.f1_score, 3)
    ])

log_less_array = []
iteration_count_array = [100, 500, 1000, 5000]
model.train_model(learning_rate=0.01, iteration_count=100, isNewtonOptimisation=False)
log_less_array.append(model.log_less)
model.train_model(learning_rate=0.01, iteration_count=500, isNewtonOptimisation=False)
log_less_array.append(model.log_less)
model.train_model(learning_rate=0.01, iteration_count=1000, isNewtonOptimisation=False)
log_less_array.append(model.log_less)
model.train_model(learning_rate=0.01, iteration_count=5000, isNewtonOptimisation=False)
log_less_array.append(model.log_less)



plt.plot(iteration_count_array, log_less_array, color='blue', lw=2)
plt.xlabel('Iteration count')
plt.ylabel('Log less')
plt.title('Precision-Recall Curve')
plt.show()

# koeffs_array = model.train_model_new(learning_rate=0.001, iterations_count=iterations_count, isNewtonOptimisation=False)
# for i in range(len(koeffs_array)) :
#     model.koeffs = koeffs_array[i]
#     model.testing_model()
#     results.append([
#         'Градиентный спуск',
#         0.001,
#         iterations_count[i],
#         round(model.log_less, 3),
#         round(model.accuracy, 3),
#         round(model.precision, 3),
#         round(model.recall, 3),
#         round(model.f1_score, 3)
#     ])

# koeffs_array = model.train_model_new(learning_rate=0.0001, iterations_count=iterations_count, isNewtonOptimisation=False)
# for i in range(len(koeffs_array)) :
#     model.koeffs = koeffs_array[i]
#     model.testing_model()
#     results.append([
#         'Градиентный спуск',
#         0.0001,
#         iterations_count[i],
#         round(model.log_less, 3),
#         round(model.accuracy, 3),
#         round(model.precision, 3),
#         round(model.recall, 3),
#         round(model.f1_score, 3)
#     ])


# koeffs_array = model.train_model_new(iterations_count=iterations_count, isNewtonOptimisation=True)
# for i in range(len(koeffs_array)) :
#     model.koeffs = koeffs_array[i]
#     model.testing_model()
#     results.append([
#         'Оптимизация Ньютона',
#         '-',
#         iterations_count[i],
#         round(model.log_less, 3),
#         round(model.accuracy, 3),
#         round(model.precision, 3),
#         round(model.recall, 3),
#         round(model.f1_score, 3)
#     ])

# Заголовки столбцов
# headers = ["Метод", "learning rate", "Число итераций" "Log Loss", "Accuracy", "Precision", "Recall", "F1 Score"]

# # Вывод таблицы
# print('\n')
# print(tabulate(results, headers, tablefmt="grid"))
