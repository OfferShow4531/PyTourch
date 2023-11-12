import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Загрузка набора данных Iris
iris = datasets.load_digits()
X = iris.data
y = iris.target

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Создание полиномиальных признаков
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Стандартизация полиномиальных данных
scaler_poly = StandardScaler()
X_train_poly = scaler_poly.fit_transform(X_train_poly)
X_test_poly = scaler_poly.transform(X_test_poly)

# Преобразование данных в тензоры PyTorch
X_train_poly = torch.Tensor(X_train_poly)
X_test_poly = torch.Tensor(X_test_poly)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Нейронная сеть с одним слоем (На основе полиномиального слоя классификации)
class DigitsClassifier(nn.Module):
    def __init__(self):
        super(DigitsClassifier, self).__init__()
        self.fc1 = nn.Linear(X_train_poly.shape[1], 64)
        self.fc2 = nn.Linear(64, 10)  # Исправлено: Выходные классы: 10

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Модель классификатора цифр
model = DigitsClassifier()

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Список для хранения значений функции потерь
losses = []

# Модель (обучение с помощью эпохи)
for epoch in range(1000):
    # Прямой проход
    outputs = model(X_train_poly)

    # Определение ошибки
    loss = criterion(outputs, y_train)

    # Обратный проход и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Добавляем значение функции потерь
    losses.append(loss.item())

# Проверяем точность модели на тестовом наборе
with torch.no_grad():
    outputs = model(X_test_poly)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = torch.sum(predicted == y_test).item() / len(y_test)
    print(f"Accuracy on test data: {accuracy}")
    print(f"Predicted on test: {predicted}")

    # Построение Шестигранных значений для визуализации связи между предсказаниями и фактическими метками
    plt.figure(figsize=(10, 8))
    plt.hexbin(predicted.numpy(), y_test.numpy(), gridsize=(15, 10), cmap='Blues')
    plt.colorbar(label='Count in Digits')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Hexbin Plot: Predicted vs True Class')
    plt.show()