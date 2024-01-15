import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
np.random.seed(3423)

def gauss(A, b, pivoting=False):
    A = A.copy()
    b = b.copy()
    n = len(b)

    for i in range(n):
        if pivoting:
            max_row = np.argmax(np.abs(A[i:, i])) + i
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros_like(b)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x

def thomas(A, b):
    n = len(b)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    
    c_prime[0] = A[0, 1] / A[0, 0]
    d_prime[0] = b[0] / A[0, 0]

    for i in range(1, n):
        a = A[i, i-1]
        d = A[i, i]
        if i < n - 1:
            c = A[i, i+1]
            c_prime[i] = c / (d - a * c_prime[i-1])
        
        d_prime[i] = (b[i] - a * d_prime[i-1]) / (d - a * c_prime[i-1])
    x = np.zeros_like(b)
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x

def generate_random_matrices(n, tridiagonal=False):
    matrices = []
    while len(matrices) < n:
        if tridiagonal:
            a = np.diag(np.random.rand(6)*2 - 1)
            b = np.diag(np.random.rand(5)*2 - 1, k=1)
            c = np.diag(np.random.rand(5)*2 - 1, k=-1)
            mat = a + b + c
        else:
            mat = np.random.rand(6, 6)*2 - 1
        
        if np.linalg.det(mat) != 0:
            matrices.append(mat.astype(np.float32))

    return matrices
    
# Generate 1000 random matrices of each type
n_matrices = 1000
random_matrices = generate_random_matrices(n_matrices)
random_tridiagonal_matrices = generate_random_matrices(n_matrices, tridiagonal=True)

# Solution vector
b = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)

def calculate_errors_squared_and_supremum(matrices, b, solution_method, universal_method=gauss):
    errors_squared = []
    errors_supremum = []
    for A in matrices:
        x_special = solution_method(A, b)
        x_universal = universal_method(A, b, pivoting=True)
        error_squared = np.linalg.norm(x_special - x_universal, ord=2) / np.linalg.norm(x_universal, ord=2)
        error_supremum = np.linalg.norm(x_special - x_universal, ord=np.inf) / np.linalg.norm(x_universal, ord=np.inf)
        errors_squared.append(error_squared)
        errors_supremum.append(error_supremum)
    return errors_squared, errors_supremum

errors_general_squared, errors_general_supremum = calculate_errors_squared_and_supremum(random_matrices, b, gauss)
errors_tridiagonal_squared, errors_tridiagonal_supremum = calculate_errors_squared_and_supremum(random_tridiagonal_matrices, b, thomas)

pdf = PdfPages("Figures.pdf")
fig = plt.figure(figsize=(14, 12))

# Ошибки для матриц общего вида (Квадратная норма)
plt.subplot(2, 2, 1)
plt.hist(errors_general_squared, bins=30, color='blue', alpha=0.7, range=(0, 1e-5))
plt.title('Ошибки для матриц общего вида (Среднеквадратичная норма)')
plt.xlabel('Относительная ошибка')
plt.ylabel('Частота')

# Ошибки для матриц общего вида (Супремум-норма)
plt.subplot(2, 2, 2)
plt.hist(errors_general_supremum, bins=30, color='red', alpha=0.7, range=(0, 1e-5))
plt.title('Ошибки для матриц общего вида (Супремум-норма)')
plt.xlabel('Относительная ошибка')

# Ошибки для трехдиагональных матриц (Квадратная норма)
plt.subplot(2, 2, 3)
plt.hist(errors_tridiagonal_squared, bins=30, color='green', alpha=0.7, range=(0, 1e-6))
plt.title('Ошибки для трехдиагональных матриц (Среднеквадратичая норма)')
plt.xlabel('Относительная ошибка')
plt.ylabel('Частота')

# Ошибки для трехдиагональных матриц (Супремум-норма)
plt.subplot(2, 2, 4)
plt.hist(errors_tridiagonal_supremum, bins=30, color='orange', alpha=0.7, range=(0, 1e-6))
plt.title('Ошибки для трехдиагональных матриц (Супремум-норма)')
plt.xlabel('Относительная ошибка')

plt.tight_layout()
pdf.savefig()
plt.show()
pdf.close()

def print_error_statistics(errors_squared, errors_supremum, matrix_type):
    print(f"Statistics for {matrix_type} Matrices:")
    print(f"  - Mean of Squared Errors: {np.mean(errors_squared)}")
    print(f"  - Median of Squared Errors: {np.median(errors_squared)}")
    print(f"  - Standard Deviation of Squared Errors: {np.std(errors_squared)}")
    print(f"  - Max Squared Error: {np.max(errors_squared)}")
    print(f"  - Mean of Supremum Errors: {np.mean(errors_supremum)}")
    print(f"  - Median of Supremum Errors: {np.median(errors_supremum)}")
    print(f"  - Standard Deviation of Supremum Errors: {np.std(errors_supremum)}")
    print(f"  - Max Supremum Error: {np.max(errors_supremum)}")
    print()

print_error_statistics(errors_general_squared, errors_general_supremum, "General")

print_error_statistics(errors_tridiagonal_squared, errors_tridiagonal_supremum, "Tridiagonal")


def generate_positive_definite_matrix(size=6):
    while True:
        A = np.random.rand(size, size) * 2 - 1
        A = (A + A.T) / 2

        A += np.eye(size) * size

        max_val = np.max(np.abs(A))
        A = A / max_val

        if np.all(np.linalg.eigvals(A) > 0):
            return A

def generate_random_matrices_extended(n, tridiagonal=False, positive_definite=False):
    matrices = []
    for _ in range(n):
        if tridiagonal:
            # Генерация трехдиагональной матрицы со значениями от -1 до 1
            a = np.diag(np.random.rand(6)*2 - 1)
            b = np.diag(np.random.rand(5)*2 - 1, k=1)
            c = np.diag(np.random.rand(5)*2 - 1, k=-1)
            mat = a + b + c
        elif positive_definite:
            # Генерация положительно определенной матрицы
            mat = generate_positive_definite_matrix()
        else:
            # Генерация случайной матрицы со значениями от -1 до 1
            mat = np.random.rand(6, 6)*2 - 1

        matrices.append(mat.astype(np.float32))

    return matrices

random_positive_matrices = generate_random_matrices_extended(n_matrices, positive_definite=True)

# Функция для разложения Холецкого и решения СЛАУ
# def cholesky(A, b):
#     n = A.shape[0]
#     L = np.zeros_like(A)

#     for i in range(n):
#         for j in range(i+1):
#             sum = 0
#             if j == i: 
#                 for k in range(j):
#                     sum += L[j, k] ** 2
#                 L[j, j] = np.sqrt(A[j, j] - sum)
#             else:
#                 for k in range(j):
#                     sum += L[i, k] * L[j, k]
#                 L[i, j] = (A[i, j] - sum) / L[j, j]

#     y = np.zeros_like(b)
#     for i in range(n):
#         sum = 0
#         for j in range(i):
#             sum += L[i, j] * y[j]
#         y[i] = (b[i] - sum) / L[i, i]

#     x = np.zeros_like(y)
#     for i in range(n-1, -1, -1):
#         sum = 0
#         for j in range(i+1, n):
#             sum += L[i, j] * x[j]
#         x[i] = (y[i] - sum) / L[i, i]

#     return x

def cholesky(A, b):
    L = np.linalg.cholesky(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x

# Вычисление ошибок для метода Холецкого и метода Гаусса
errors_cholesky_squared, errors_cholesky_supremum = calculate_errors_squared_and_supremum(random_positive_matrices, b, cholesky)

# Построение гистограмм для сравнения методов
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Ошибки для метода Холецкого (Квадратная норма)
axs[0].hist(errors_cholesky_squared, bins=30, color='blue', alpha=0.7)
axs[0].set_title('Ошибки для метода Холецкого (Среднеквадратичная норма)')
axs[0].set_xlabel('Относительная ошибка')
axs[0].set_ylabel('Частота')

# Ошибки для метода Холецкого (Супремум-норма)
axs[1].hist(errors_cholesky_supremum, bins=30, color='red', alpha=0.7)
axs[1].set_title('Ошибки для метода Холецкого (Супремум-норма)')
axs[1].set_xlabel('Относительная ошибка')

plt.tight_layout()
pdf = PdfPages("Cholesky_Errors.pdf")
pdf.savefig(fig)
plt.show()
pdf.close()


# Функция для вычисления спектральных радиусов и чисел обусловленности матриц
def calculate_spectral_radius_and_condition_number(matrices):
    spectral_radii = []
    condition_numbers = []
    for A in matrices:
        eigenvalues = np.linalg.eigvals(A)
        spectral_radius = max(abs(eigenvalues))
        condition_number = np.linalg.cond(A)
        spectral_radii.append(spectral_radius)
        condition_numbers.append(condition_number)
    return spectral_radii, condition_numbers

# Вычисление спектральных радиусов и чисел обусловленности для каждого типа матриц
spectral_radius_random, condition_number_random = calculate_spectral_radius_and_condition_number(random_matrices)
spectral_radius_tridiagonal, condition_number_tridiagonal = calculate_spectral_radius_and_condition_number(random_tridiagonal_matrices)
spectral_radius_positive, condition_number_positive = calculate_spectral_radius_and_condition_number(random_positive_matrices)

# max_cond = 0
# save = 0
# for i in range(1000):
#     if max_cond < condition_number_random[i]:
#         max_cond = condition_number_random[i]
#         save = i

# print(save)
# print(spectral_radius_random[save], condition_number_random[save])
# print(random_matrices[save])

# Построение гистограмм
fig, axs = plt.subplots(3, 2, figsize=(14, 18))

# Спектральные радиусы общих матриц
axs[0, 0].hist(spectral_radius_random, bins=30, color='blue', alpha=0.7)
axs[0, 0].set_title('Спектральные радиусы общих матриц')
axs[0, 0].set_xlabel('Спектральный радиус')
axs[0, 0].set_ylabel('Частота')

# Числа обусловленности общих матриц
axs[0, 1].hist(condition_number_random, bins=30, color='red', alpha=0.7, range=(0, 100))
axs[0, 1].set_title('Числа обусловленности общих матриц')
axs[0, 1].set_xlabel('Число обусловленности')

# Спектральные радиусы трехдиагональных матриц
axs[1, 0].hist(spectral_radius_tridiagonal, bins=30, color='green', alpha=0.7)
axs[1, 0].set_title('Спектральные радиусы трехдиагональных матриц')
axs[1, 0].set_xlabel('Спектральный радиус')
axs[1, 0].set_ylabel('Частота')

# Числа обусловленности трехдиагональных матриц
axs[1, 1].hist(condition_number_tridiagonal, bins=30, color='orange', alpha=0.7, range=(0, 100))
axs[1, 1].set_title('Числа обусловленности трехдиагональных матриц')
axs[1, 1].set_xlabel('Число обусловленности')

# Спектральные радиусы положительно определенных матриц
axs[2, 0].hist(spectral_radius_positive, bins=30, color='purple', alpha=0.7)
axs[2, 0].set_title('Спектральные радиусы положительно определенных матриц')
axs[2, 0].set_xlabel('Спектральный радиус')
axs[2, 0].set_ylabel('Частота')

# Числа обусловленности положительно определенных матриц
axs[2, 1].hist(condition_number_positive, bins=30, color='brown', alpha=0.7)
axs[2, 1].set_title('Числа обусловленности положительно определенных матриц')
axs[2, 1].set_xlabel('Число обусловленности')

plt.tight_layout()
pdf = PdfPages("Spectral.pdf")
pdf.savefig(fig)
plt.show()
pdf.close()