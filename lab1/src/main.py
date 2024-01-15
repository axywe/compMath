import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pdf_pages = PdfPages('plots.pdf')
data = np.loadtxt("contour.txt")
x = data[:, 0]
y = data[:, 1]

fig1 = plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=5, color='indigo', label='Точки контура')
plt.title("Начальное положение фрактала")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
pdf_pages.savefig(fig1, bbox_inches='tight')

M = 10
N = len(data)
hat_N = (N + M) // M 
indices = [M * i for i in range(hat_N) if M * i < N]

x_hat = x[indices]
y_hat = y[indices]

fig2 = plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=5, color='indigo', label='Точки контура')
plt.scatter(x_hat, y_hat, s=40, color='red', marker='o', label='Точки интерполяции')
plt.title("Точки интерполяции")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
pdf_pages.savefig(fig2, bbox_inches='tight')

def gauss(A, B):
    n = len(A)
    
    AB = []
    for i in range(n):
        AB.append(list(A[i]) + [B[i]])

    for i in range(n):
        pivot_row = i
        max_val = abs(AB[i][i])
        for k in range(i + 1, n):
            if abs(AB[k][i]) > max_val:
                max_val = abs(AB[k][i])
                pivot_row = k
                
        AB[i], AB[pivot_row] = AB[pivot_row], AB[i]

        for j in range(i + 1, n):
            factor = AB[j][i] / AB[i][i]
            for k in range(i, n + 1):
                AB[j][k] -= factor * AB[i][k]

    X = [0] * n
    for i in range(n - 1, -1, -1):
        X[i] = AB[i][-1]
        for j in range(i + 1, n):
            X[i] -= AB[i][j] * X[j]
        X[i] /= AB[i][i]

    return X

def coefficients(t, y):
    n = len(t)
    
    h = t[1] - t[0]

    delta_y = [y[i+1] - y[i] for i in range(n-1)]
        
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[n-1, n-1] = 1
    for i in range(1, n-1):
        A[i, i-1] = h
        A[i, i] = 2 * (h + h)
        A[i, i+1] = h
    
    B = np.zeros(n)
    for i in range(1, n-1):
        B[i] = 3 * (delta_y[i] / h - delta_y[i-1] / h)
    C = gauss(A, B)
    
    a = y[:-1]
    
    b = [delta_y[i] / h - h * (2*C[i] + C[i+1]) / 3 for i in range(n-1)]
    
    d = [(C[i+1] - C[i]) / (3 * h) for i in range(n-1)]
    
    coefficients = np.column_stack((a, b, C[:-1], d))
    
    return coefficients

def I_j(t, t_j, j):
    if t >= t_j[j] and t < t_j[j + 1]:
        return 1
    else:
        return 0

def interpolate(t, t_j, coeffs):
    n = len(coeffs)
    
    for j in range(n):
        if j == n - 1 or I_j(t, t_j, j):
            delta_t = t - t_j[j]
            a, b, c, d = coeffs[j]
            return a + b*delta_t + c*delta_t**2 + d*delta_t**3
    return 0

a_jk = np.array(coefficients(indices, x_hat))
b_jk = np.array(coefficients(indices, y_hat))

h = 0.1

t_values = np.arange(0, N, h)

x_spline = [interpolate(t, indices, a_jk) for t in t_values]
y_spline = [interpolate(t, indices, b_jk) for t in t_values]

fig3 = plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=5, color='indigo', label='Точки контура')
plt.scatter(x_hat, y_hat, s=10, color='red', label='Точки интерполяции')
plt.plot(x_spline, y_spline, color='darkorange', label='Кубический сплайн')
plt.title('Визуализация кубических сплайнов')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
pdf_pages.savefig(fig3, bbox_inches='tight')

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

distances = [distance(x[i], y[i], interpolate(i, indices, a_jk), interpolate(i, indices, b_jk)) for i in range(N)]

fig4 = plt.figure(figsize=(10, 6))
plt.bar(range(N), distances, width=1, color='#0ea27e',
        label=f'Среднее расстояние: {np.mean(distances)}\nСтандартное отклонение: {np.std(distances)}')
plt.title('Гистограмма изменения расстояния между двумя точками')
plt.xlabel('Номер точки')
plt.ylabel('Расстояние')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
pdf_pages.savefig(fig4, bbox_inches='tight')

distances_interpolation = distances[:-((N%10)-1)]
fig5 = plt.figure(figsize=(10, 6))
plt.bar(range(N-(N%10)+1), distances_interpolation, width=1, color='#0ea27e',
        label=f'Среднее расстояние: {np.mean(distances_interpolation)}\nСтандартное отклонение: {np.std(distances_interpolation)}')
plt.title('Гистограмма изменения расстояния между двумя точками')
plt.xlabel('Номер точки')
plt.ylabel('Расстояние')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
pdf_pages.savefig(fig5, bbox_inches='tight')

def lab1_base(filename_in:str, factor:int, filename_out:str):
    data = np.loadtxt(filename_in)
    x = data[:, 0]
    y = data[:, 1]
    N = len(data)
    hat_N = (N + factor) // factor
    indices = [factor * i for i in range(hat_N) if factor * i < N]
    
    x_hat = x[::factor]
    y_hat = y[::factor]
    t_hat = np.arange(len(x_hat))
    
    coeffs_x = coefficients(t_hat, x_hat)
    coeffs_y = coefficients(t_hat, y_hat)

    a_jk = np.array(coefficients(indices, x_hat))
    b_jk = np.array(coefficients(indices, y_hat))

    h = 0.1

    t_values = np.arange(0, N, h)

    x_spline = [interpolate(t, indices, a_jk) for t in t_values]
    y_spline = [interpolate(t, indices, b_jk) for t in t_values]

    coeffs = np.concatenate((coeffs_x, coeffs_y), axis=1)
    np.savetxt(filename_out, coeffs, delimiter='\t')

    distances = [distance(x[i], y[i], interpolate(i, indices, a_jk), interpolate(i, indices, b_jk)) for i in range(N)]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=5, color='indigo', label='Точки контура')
    plt.scatter(x_hat, y_hat, s=10, color='red', label='Точки интерполяции')
    plt.plot(x_spline, y_spline, color='darkorange', label='Кубический сплайн')
    plt.plot([], [], ' ', label=f'Среднее расстояние: {np.mean(distances)}\nСтандартное отклонение: {np.std(distances)}')
    plt.title('Визуализация кубических сплайнов')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()
    return coeffs

coeffs = lab1_base("contour.txt", N-1, "coeffs.txt")

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

class AutoDiffNum:
    def __init__(self, real, dual=0.0):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self.real + other.real, self.dual + other.dual)
        return AutoDiffNum(self.real + other, self.dual)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self.real - other.real, self.dual - other.dual)
        return AutoDiffNum(self.real - other, self.dual)

    def __mul__(self, other):
        if isinstance(other, AutoDiffNum):
            return AutoDiffNum(self.real * other.real, self.real * other.dual + self.dual * other.real)
        return AutoDiffNum(self.real * other, self.dual * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, AutoDiffNum):
            real_part = self.real / other.real
            dual_part = (self.dual - real_part * other.dual) / other.real
            return AutoDiffNum(real_part, dual_part)
        return AutoDiffNum(self.real / other, self.dual / other)

    def __pow__(self, power):
        if power == 0:
            return AutoDiffNum(1)
        if power == 1:
            return self
        real_part = self.real ** power
        dual_part = power * self.real ** (power - 1) * self.dual
        return AutoDiffNum(real_part, dual_part)

    def __repr__(self):
        return f"{self.real} + ε{self.dual}"

    @staticmethod
    def I_j(t, t_j, j):
        return t >= t_j[j] and t < t_j[j + 1]

    @staticmethod
    def interpolate_with_derivative(t, t_j, coeffs):
        def I_j(t_real, t_j, j):
            return t_real >= t_j[j] and t_real < t_j[j + 1]
        
        n = len(coeffs)
        for j in range(n):
            if j == n - 1 or I_j(t.real, t_j, j):
                delta_t = t - AutoDiffNum(t_j[j])
                a, b, c, d = coeffs[j]
                return a + b * delta_t + c * delta_t ** 2 + d * delta_t ** 3

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def calculate_normals(x, y):
    return -y, x

fig6, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, s=5, color='indigo', label='Точки контура')
ax.scatter(x_hat, y_hat, s=40, color='red', label='Точки разреженного множества')
plt.plot(x_spline, y_spline, color='darkorange', label='Кубический сплайн')

tangents = []
normals = []

for t in indices[1::3]:
    t_diff = AutoDiffNum(t, 1)
    x_t = AutoDiffNum.interpolate_with_derivative(t_diff, indices, a_jk)
    y_t = AutoDiffNum.interpolate_with_derivative(t_diff, indices, b_jk)
    
    tangent = (x_t.dual, y_t.dual)
    tangents.append(tangent)

    
    normal = calculate_normals(x_t.dual, y_t.dual)
    normals.append(normal)

scale_factor = 0.0004

tangent_x, tangent_y = zip(*[(dx * scale_factor, dy * scale_factor) for dx, dy in tangents])
normal_x, normal_y = zip(*[(dx * scale_factor, dy * scale_factor) for dx, dy in normals])

normalized_tangents = [normalize_vector(np.array(tangent)) for tangent in tangents]
normalized_normals = [normalize_vector(np.array(normal)) for normal in normals]

normalized_tangent_x, normalized_tangent_y = zip(*[(dx * scale_factor, dy * scale_factor) for dx, dy in normalized_tangents])
normalized_normal_x, normalized_normal_y = zip(*[(dx * scale_factor, dy * scale_factor) for dx, dy in normalized_normals])

ax.quiver(x_hat[1::3], y_hat[1::3], normalized_tangent_x, normalized_tangent_y, 
          angles='xy', scale_units='xy', scale=0.5, color='blue', label='Вектор')
ax.quiver(x_hat[1::3], y_hat[1::3], normalized_normal_x, normalized_normal_y, 
          angles='xy', scale_units='xy', scale=0.5, color='green', label='Нормаль')

plt.title('Векторы и нормали к кубическому сплайну')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
pdf_pages.savefig(fig6, bbox_inches='tight')

def distance_to_point(t, xi, yi, a_jk, b_jk, indices):
    x_t = interpolate(t, indices, a_jk)
    y_t = interpolate(t, indices, b_jk)
    return distance(xi, yi, x_t, y_t)

def minimize(func, a, b, tol=1e-5, max_iter=10, *args):
    phi = (1 + 5 ** 0.5) / 2
    iter_count = 0
    while iter_count < max_iter and (b - a) > tol:
        c1 = b - (b - a) / phi
        c2 = a + (b - a) / phi

        f_c1 = func(c1, *args)
        f_c2 = func(c2, *args)

        if f_c1 < f_c2:
            b = c2
        else:
            a = c1

        iter_count += 1

    return (a + b) / 2

closest_points = []
closest_distances = []

for i in range(N-(N%10)):
    xi, yi = x[i], y[i]
    t_star = minimize(distance_to_point, 0, N-(N%10), 1e-5, 10, xi, yi, a_jk, b_jk, indices)
    closest_point = [interpolate(t_star, indices, a_jk), interpolate(t_star, indices, b_jk)]
    closest_points.append(closest_point)
    closest_distances.append(distance_to_point(t_star, xi, yi, a_jk, b_jk, indices))

closest_points = np.array(closest_points)

average_error = np.mean(closest_distances)

fig7 = plt.figure(figsize=(10, 6))
plt.scatter(x, y, s=5, color='indigo', label='Точки контура')
plt.scatter(x_hat, y_hat, s=10, color='red', label='Точки интерполяции')
plt.plot(x_spline, y_spline, color='darkorange', label='Кубический сплайн')
plt.scatter(closest_points[:, 0], closest_points[:, 1], s=10, color='green', label='Ближайшие точки')
plt.plot([], [], ' ', label=f'Среднее расстояние: {np.mean(closest_distances)}\nСтандартное отклонение: {np.std(closest_distances)}')
plt.title(f'Ближайшие точки на сплайне')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
pdf_pages.savefig(fig7, bbox_inches='tight')

pdf_pages.close()