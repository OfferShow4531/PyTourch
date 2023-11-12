import torch
import numpy as np
# Создаение матрицы
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# Умножение матриц
C = torch.mm(A, B)
print("Результат умножения матриц A и B:")
print(C)

# Сложение матриц
D = A + B
print("\nРезультат сложения матриц A и B:")
print(D)

# Транспонирование матрицы A
A_transposed = torch.transpose(A, 0, 1)
print("\nТранспонированная матрица A:")
print(A_transposed)




# Создание матрицы x с использованием torch.Tensor
x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Создание единичной матрицы с использованием torch.eye
identity_matrix = torch.eye(3)

# Создание матрицы из numpy массива с использованием torch.from_numpy
numpy_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix_from_numpy = torch.from_numpy(numpy_array)

# Создание матрицы из единиц с использованием torch.ones
ones_matrix = torch.ones(3, 3)

# Создание матрицы нулей с использованием torch.zeros_like
zeros_like_matrix = torch.zeros_like(ones_matrix)

# Создание тензора с использованием torch.arange
arange_tensor = torch.arange(0, 10, 2)

# Сложение двух матриц с использованием torch.add
sum_matrix = torch.add(x, identity_matrix)

# Умножение двух матриц с использованием torch.mm
product_matrix = torch.mm(x, identity_matrix)

# Поэлементное умножение двух матриц с использованием torch.mul
elementwise_product = torch.mul(x, identity_matrix)

# Экспоненциальная функция с использованием torch.exp
exp_matrix = torch.exp(x)

# Возведение в степень с использованием torch.pow
power_matrix = torch.pow(x, 2)

# Квадратный корень с использованием torch.sqrt
sqrt_matrix = torch.sqrt(power_matrix)

# Сигмоидальная функция с использованием torch.sigmoid
sigmoid_matrix = torch.sigmoid(x)

# Накопление произведения элементов с использованием torch.cumprod
cumulative_product = torch.cumprod(arange_tensor, dim=0)

# Сумма всех элементов с использованием torch.sum
total_sum = torch.sum(x)

# Стандартное отклонение с использованием torch.std
standard_deviation = torch.std(x)

# Среднее значение с использованием torch.mean
mean_value = torch.mean(x)

# Создание переменной с использованием torch.autograd.Variable
x_variable = torch.autograd.Variable(torch.ones(4, 4), requires_grad=True)

# Вывод результатов
print("Matrix x:")
print(x)
print("\nIdentity Matrix:")
print(identity_matrix)
print("\nMatrix from numpy array:")
print(matrix_from_numpy)
print("\nOnes Matrix:")
print(ones_matrix)
print("\nZeros Matrix (like ones_matrix):")
print(zeros_like_matrix)
print("\nArange Tensor:")
print(arange_tensor)
print("\nSum of x and Identity Matrix:")
print(sum_matrix)
print("\nProduct of x and Identity Matrix:")
print(product_matrix)
print("\nElementwise Product of x and Identity Matrix:")
print(elementwise_product)
print("\nExponential of x:")
print(exp_matrix)
print("\nPower of x (raised to the power of 2):")
print(power_matrix)
print("\nSquare Root of Power Matrix:")
print(sqrt_matrix)
print("\nSigmoid of x:")
print(sigmoid_matrix)
print("\nCumulative Product of Arange Tensor:")
print(cumulative_product)
print("\nTotal Sum of x:")
print(total_sum)
print("\nStandard Deviation of x:")
print(standard_deviation)
print("\nMean Value of x:")
print(mean_value)
print("\nVariable x:")
print(x_variable)