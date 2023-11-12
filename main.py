import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка набора данных Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Преобразование данных в тензоры PyTorch
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test) #Вычислений чисел (16 бит integer)
y_train = torch.LongTensor(y_train) #Вычислений больших целых чисел (64 бит integer)
y_test = torch.LongTensor(y_test)

# Нейронная сеть с одним слоем (На основе линейного слоя класификации)
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 4)  # Входные признаки: 4, Выходные классы: 4

    def forward(self, x):
        x = self.fc1(x)
        return x

# Модель
model = IrisClassifier()

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Список для хранения значений функции потерь
losses = []

# Модель (обучение с помощью эпохи)
for epoch in range(1000):
    # Прямой проход
    outputs = model(X_train)

    # Определение ошибки
    loss = criterion(outputs, y_train)

    # Обратный проход и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Добавляем значение функции потерь в список
    losses.append(loss.item())

# Проверяем точность модели на тестовом наборе
with torch.no_grad():
    outputs = model(X_test)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = torch.sum(predicted == y_test).item() / len(y_test)
    print(f"Accuracy on test data: {accuracy}")
    print(f"Predicted on test: {predicted}")

    # Вывод значения точности на графике (Легенда)
    plot_accuracy = mpatches.Patch(color='red', label=f"accuracy: ${accuracy}")

    # Построение графика функции потерь
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend(handles=[plot_accuracy])
    plt.show()