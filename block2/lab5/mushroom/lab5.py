import pandas as pd
import numpy as np
import graphviz
import math
import matplotlib.pyplot as plt


class Result :
    '''
    класс результата проверки теста
    count -- количество результатов
    p_count -- количество положительных результатов
    real_result -- реальный результат
    '''
    def __init__(self, count = None, p_count = None, real_result = None) :
        self.count = count
        self.p_count = p_count
        self.expected = 1 if real_result == 'e' else 0


class Node :
    """
    attributes -- массив, внутри которого словарь
    isList == True -- Node является листом. В противном случае Node является узлом
    parent -- ссылка на родительский элемент. Если parent is None, Node является корнем дерева
    children -- словарь. key -- значения атрибутов, values -- ссылки на дочерние Node
    """
    node_count = 0
    def __init__(self, attributes, parent = None) :
        self.id = Node.node_count
        Node.node_count += 1
        self.attributes = attributes
        self.attribute = None
        self.node_level = 0
        self.children = {}
        self.parent = parent
        self.isList = True
        self.class_count = len(attributes)
        self.positive_class_count = sum(1 for attribute in attributes if attribute['class'] == 'e')

    def update(self) :
        self.class_count = len(self.attributes)
        self.positive_class_count = sum(1 for attribute in self.attributes if attribute['class'] == 'e')

class Tree :
    """
    root -- корневой узел
    current_node -- текущий узел
    """
    def __init__(self, root) :
        self.root = root
        self.current_node = self.root
        self.last_node_level = 0

    def __delete_current_node__ (self) :
        '''
        удаление узла, на котором дерево находится в данный момент
        '''
        if self.current_node == self.root :
            return False
        if len(self.current_node.parent.children) > 1 :
            return False

        self.current_node.parrent.children = self.current_node.children
        for node in self.current_node.children.values() :
            node.parent = self.current_node.parrent
        self.current_node = self.current_node.parent

        return True

    def go_to_next_node_by_attribute_value(self, attribute_value) :
        '''
        переход на следующий узел по заданному значению атрибута
        если удалось перейти, возвращает True. В противном случае возвращает False
        '''
        if attribute_value not in self.current_node.children.keys() : return False
        self.current_node = self.current_node.children[attribute_value]
        return True

    def go_to_prev_node(self) :
        '''
        переход на предыдущий узел
        если удалось перейти, возвращает True. В противном случае возвращает False
        '''
        if self.current_node.parent is None : return False
        self.current_node = self.current_node.parent
        return True

    def go_to_root(self) :
        '''
        переход в корневой узел
        '''
        self.current_node = self.root
    
    def train_the_model(self, attributes_priority) :
        '''
        обучение модели
        '''
        priority = attributes_priority[:]
        while len(priority) > 0 :
            self.__add_next_level__(priority.pop())

    def __add_next_level__(self, attribute, node = None) :
        '''
        добавление следующего уровня
        '''
        node = node if node is not None else self.root

        if node.node_level == self.last_node_level and node.isList :
            self.__add_next_nodes_by_attribute__(attribute, node)
        else :
            for child in node.children.values() :
                self.__add_next_level__(attribute, child)

        if node.node_level == 0 :
            self.last_node_level += 1

    def __add_next_nodes_by_attribute__(self, attribute, node) :
        '''
        Добавление детей для конкретного узла
        '''
        node.attribute = attribute
        for atributes_arr in node.attributes :
            key = atributes_arr[attribute]
            if key not in node.children.keys() :
                node.children[key] = Node([])
            node.children[key].attributes.append(atributes_arr)
            node.children[key].parent = node
            node.children[key].node_level = node.node_level + 1
            node.children[key].update()
        node.attributes = None
        if node.children is not None and len(node.children) != 0 : node.isList = False

    def print_tree(self, node = None) :
        node = node if node is not None else self.root

        print(node.node_level, ' | ', node.children.keys())
        for child in node.children.values() :
            self.print_tree(child)

    def generatr_tree_png(self, node = None, dot = None) :
        node = node if node is not None else self.root
        dot = dot if dot is not None else graphviz.Digraph(comment='Tree')
        title = node.attribute if node.attribute is not None else node.positive_class_count / node.class_count
        dot.node(str(node.id), str(title))

        if node.isList : return

        for key in node.children.keys() :
            self.generatr_tree_png(node.children[key], dot)
            dot.edge(str(node.id), str(node.children[key].id), key)
        
        if node.node_level == 0 :
            dot.render('tree', view=True)


def get_subcolumns(array1, array2):
    '''
    Разбивает array 2 на подмассивы соответственно с атрибутами array1
    '''
    grouped_array = {}

    for i in range(len(array1)):
        key = array2[i]
        if key not in grouped_array:
            grouped_array[key] = []
        grouped_array[key].append(array1[i])

    # Преобразуем словарь в список подмассивов
    result = list(grouped_array.values())

    return result


def entropy(column):
    '''
    энтропия
    '''
    _, counts = np.unique(column, return_counts=True) 
    frequency = counts / counts.sum() 
    entropy = -np.sum(frequency * np.log2(frequency)) 
    return entropy 


def information_gain_criterion(column, columns) : 
    ''' 
    критерий прироста информации 
    '''
    main_entropy = entropy(column) 
    entropies = np.array([entropy(subcolumn) for subcolumn in columns]) 
    subcounts = np.array([len(subcolumn) for subcolumn in columns]) 
    return main_entropy - np.sum((subcounts / len(column)) * entropies) 


def metrics(results, threshold) :
    '''
    Считает параметры
        true_positive
        false_positive
        true_negative
        false_negative
    В зависимости от правила размещения
    '''
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for result in results :
        p_probability = result.p_count / result.count
        flag = p_probability >= threshold
        if flag > threshold :
            if result.expected == 1 :
                true_positive += 1
            else :
                false_positive += 1
        else :
            if result.expected == 0 :
                true_negative += 1
            else :
                false_negative += 1

    return true_positive, false_positive, true_negative, false_negative


def roc_args(results) :
    '''
    метод для подсчёта TPR и FPR при различных порогах для построения AUC-ROC кривой
    '''
    tpr = []
    fpr = []
    thresholds = np.arange(0.0, 1.01, 0.01)
    for threshold in thresholds :
        tp, fp, tn, fn = metrics(results, threshold)
        current_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        current_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr.append(current_tpr)
        fpr.append(current_fpr)

    return tpr, fpr


def pr_args(results) :
    precisions = []
    recalls = []
    thresholds = np.arange(0.0, 1.01, 0.01)
    for threshold in thresholds :
        tp, fp, _, fn = metrics(results, threshold)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


'''
Чтение данных из файла
'''
file_path = 'mushroom//agaricus-lepiota.csv'
data = pd.read_csv(file_path)


'''
заменяем "?" на моду
'''
for column in data.columns:
    mode_value = data[column].mode()[0] 
    data[column] = data[column].replace('?', mode_value)


'''
выбираем рандомные признаки
'''
number_of_features = round(math.sqrt(len(data.iloc[0]) - 1))
np.random.seed(23)
feature_columns = data.columns.drop('class')
selected_features = np.random.choice(feature_columns, size=number_of_features, replace=False)
print("Случайно отобранные признаки: " + str(selected_features))
print()
print(data[selected_features].describe())
print()


'''
Разделение данных на обучающую и тестовую выборку в соотношении 80/20
'''
new_data = data[['class'] + list(selected_features)]
shuffled_indices = np.random.permutation(new_data.index)
train_size = int(0.8 * len(new_data))
test_indices = shuffled_indices[train_size:]
train_indices = shuffled_indices[:train_size]

train_data = new_data.loc[train_indices]
test_data = new_data.loc[test_indices]


'''
Получение обучающей и тестовой выборки
'''
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']


'''
Подсчёт коэффициента прироста информации к каждой колонке по отношению к колонке class
koeffs -- массив с коэффициентами прироста информации
features -- массив колонок
'''
features = []
koeffs = []
result = y_train.to_numpy()
X_train = X_train.reset_index(drop=True)
for data_name in selected_features :
    features.append(X_train[data_name].to_numpy())
    columns = get_subcolumns(result, features[-1])
    koeffs.append(information_gain_criterion(result, columns))
    print('Критерий прироста информации ', round(koeffs[-1], 3), ' для ', data_name)
print()


'''
Сортировка по возрастанию критерия прироста информации
сами koeffs мы не сортируем, чтобы потом по ним же отсортировать тестовые данные
'''
combined_sorted = sorted(zip(koeffs, features, selected_features), key=lambda x: x[0], reverse=False)
_, features, selected_features = map(list, zip(*combined_sorted))


'''
Заполняем данные `attributes`, чтобы в дальнейшем положить в `Node` и обучить `Tree`
Как должны выглядеть данные:
```
[
{'attribute1': 'param1', 'attribute2': 'param2', ..., 'attributeN': 'paramN,'class': 'e / p',}
{'attribute1': 'param1', 'attribute2': 'param2', ..., 'attributeN': 'paramN,'class': 'e / p',}
{'attribute1': 'param1', 'attribute2': 'param2', ..., 'attributeN': 'paramN,'class': 'e / p',}
...
{'attribute1': 'param1', 'attribute2': 'param2', ..., 'attributeN': 'paramN,'class': 'e / p',}
]
```
'''
y_train = y_train.reset_index(drop=True)
attributes = []
for i in range(len(y_train)) :
    attribute = {}
    for j in range(len(selected_features)) :
        attribute[selected_features[j]] = features[j][i]
    attribute['class'] = y_train[i]
    attributes.append(attribute)


'''
Создание корневого Node и обучение дерева
'''
root = Node(attributes)
tree = Tree(root)
tree.train_the_model(selected_features)
# tree.generatr_tree_png()


'''
подготовка тестовых данных
'''
y_test = y_test.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
features_test = []
for data_name in reversed(selected_features) :
    features_test.append(X_test[data_name])


'''
Проверка модели
'''
results = []
for i in range(len(y_test)) :
    result = None
    tree.go_to_root()
    for j in range(len(features_test)) :
        tree.go_to_next_node_by_attribute_value(features_test[j][i])
    
    result = Result(tree.current_node.class_count, tree.current_node.positive_class_count, y_test[i])
    results.append(result)


'''
подсчёт и вывод параметров Accuracy, Precision, Recall при пороге правильного ответа 0.5
'''
print()
tp, fp, tn, fn = metrics(results, 0.5)
print('Accuracy: ' + str((tp + tn) / len(results)))
print("Precision: " + str(tp / (tp + fp)))
print("Recall: " + str(tp / (tp + fn)))


fig, axs = plt.subplots(1, 2, figsize=(12, 5))
'''
построение AUC-ROC кривой на основе TPR и FPR при разных порогах
'''
tpr, fpr = roc_args(results)
axs[0].plot(fpr, tpr, color='blue', lw=2)
axs[0].plot([0, 1], [0, 1], color='red', linestyle='--')
axs[0].set_xlim([-0.2, 1.2])
axs[0].set_ylim([-0.2, 1.2])
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC Curve')


'''
построение AUC-PR кривой на основе TPR и FPR при разных порогах
'''
precisions, recalls = pr_args(results)
axs[1].plot(recalls, precisions, color='blue', lw=2)
axs[1].set_xlim([-0.2, 1.2])
axs[1].set_ylim([-0.2, 1.2])
axs[1].set_xlabel('Recall')
axs[1].set_ylabel('Precision')
axs[1].set_title('Precision-Recall Curve')

plt.tight_layout()
plt.show()