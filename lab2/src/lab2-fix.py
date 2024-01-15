#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize


g = 9.8        
C = 1.03439984

def xt(t, x):
    return C * (t - 0.5 * np.sin(2 * t)) - x

tx = np.vectorize(lambda x: fsolve(xt, 0, x)[0])

def y(x):
    t = tx(x)
    return C * (0.5 - 0.5 * np.cos(2 * t))

def dyx(x, dx=0.001):
    return (y(x + dx) - y(x - dx)) / (2 * dx)

def Fy(x):
    return np.sqrt((1 + (dyx(x) ** 2)) / (2 * g * y(x)))
    
def composite_simpson(a, b, n, f):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    result = y[0] + y[-1]
    result += 2 * np.sum(y[2:-1:2])
    result += 4 * np.sum(y[1::2])

    return (h / 3) * result

def composite_trapezoid(a, b, n, f):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)

    result = y[0] + y[-1]
    result += 2 * np.sum(y[1:-1])

    return (h / 2) * result

with PdfPages('figure1.pdf') as pdf:
    plt.figure()
    xp = np.linspace(0.001, 2, 1000)
    yp = [y(x_) for x_ in xp]
    plt.plot(xp, yp)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Брахистохрона')
    pdf.savefig() 
    plt.show()
    plt.close()


# In[19]:


a = 0.1
b = 2
n_values = np.arange(3, 10000, 100) 
errors_simpson = []
errors_trapezoid = []

absol = composite_simpson(a, b, 11000, Fy) 

for n in n_values:
    approx_simpson = composite_simpson(a, b, n, Fy)
    approx_trapezoid = composite_trapezoid(a, b, n, Fy)
    errors_simpson.append(np.abs(approx_simpson - absol))
    errors_trapezoid.append(np.abs(approx_trapezoid - absol))

h_values = (b - a) / n_values
expected_error_simpson = h_values**4
expected_error_trapezoid = h_values**2

with PdfPages('figure2.pdf') as pdf:
    plt.figure()
    plt.loglog(n_values, errors_simpson, 'o', label='Метод Симпсона')
    plt.loglog(n_values, errors_trapezoid, 'o', label='Метод трапеций')
    plt.loglog(n_values, expected_error_simpson, 'k-', label='Ожидаемый результат метода Симпсона $h^4$')
    plt.loglog(n_values, expected_error_trapezoid, 'k--', label='Ожидаемый результат метода трапеций $h^2$')
    plt.xlabel('Количество узлов (n)')
    plt.ylabel('Абсолютная погрешность')
    plt.title('Погрешность при численном интегрировании')
    plt.legend()
    pdf.savefig()
    plt.show()
    plt.close()

def find_optimal_step_index(errors):
    threshold = 1e-6 
    for i in range(1, len(errors)):
        if np.abs(errors[i] - errors[i - 1]) < threshold:
            return i
    return len(errors) - 1 

optimal_index_simpson = find_optimal_step_index(errors_simpson)
optimal_index_trapezoid = find_optimal_step_index(errors_trapezoid)

optimal_n_simpson = n_values[optimal_index_simpson]
optimal_n_trapezoid = n_values[optimal_index_trapezoid]
optimal_h_simpson = (b - a) / optimal_n_simpson
optimal_h_trapezoid = (b - a) / optimal_n_trapezoid

optimal_n_simpson, optimal_h_simpson, optimal_n_trapezoid, optimal_h_trapezoid


# In[22]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

C = 1.03439984
T = 1.75418438
a = 2
y_a = 1
g = 9.81 

def cycloid(t):
    return C * np.array([t - 0.5 * np.sin(2 * t), 0.5 - 0.5 * np.cos(2 * t)])

def piecewise_linear_interpolation(t_points):
    t_values = np.linspace(0, T, t_points)
    cycloid_points = cycloid(t_values)

    min_x, max_x = np.min(cycloid_points[0]), np.max(cycloid_points[0])
    x_values = np.linspace(min_x, max_x, 1000)

    linear_interp_func = interp1d(cycloid_points[0], cycloid_points[1], kind='linear', fill_value="extrapolate")

    return linear_interp_func, x_values
    
def composite_simpson(a, b, n, f):
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    result = y[0] + y[-1]
    result += 2 * np.sum(y[2:-1:2])
    result += 4 * np.sum(y[1::2])

    return (h / 3) * result

def calculate_functional(interp_points, integration_points):
    interp_func, x_values = piecewise_linear_interpolation(interp_points)


    def integrand(x):
        y = interp_func(x)
        dy_dx = np.gradient(y, x)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sqrt((1 + dy_dx**2) / (2 * g * y))


    start_x = x_values[1] 
    end_x = x_values[-1]

    functional_value = composite_simpson(start_x, end_x, integration_points, integrand)
    return functional_value

interp_steps_range = np.array([10, 20, 50, 100, 200, 500, 1000])
integration_steps_range = np.logspace(-3, 0, 50) * 1000  

high_precision_interp_points = 500
high_precision_integration_points = 10000
exact_functional_value = calculate_functional(high_precision_interp_points, high_precision_integration_points)

errors = np.zeros((len(interp_steps_range), len(integration_steps_range)))
interp_steps_array, integration_steps_array = np.meshgrid(interp_steps_range, integration_steps_range, indexing='ij')

for i, interp_points in enumerate(interp_steps_range):
    for j, integration_points in enumerate(integration_steps_range):
        approx_functional_value = calculate_functional(interp_points, int(integration_points))
        errors[i, j] = np.abs(approx_functional_value - exact_functional_value)

error_log_ticks = np.linspace(np.log10(errors.min()), np.log10(errors.max()), num=5)
error_log_labels = [f"$10^{{{{{int(t)}}}}}$" for t in error_log_ticks]

with PdfPages('figure3.pdf') as pdf:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.log10(interp_steps_array), np.log10(integration_steps_array), np.log10(errors), cmap='viridis')
    ax.view_init(elev=30, azim=45) 
    
    ax.set_xlabel('Шаги интерполяции')
    ax.set_ylabel('Шаги интегрирования')
    ax.set_zlabel('L2 ошибка')
    ax.set_title('Анализ ошибки аппроксимации функционала для\n различных комбинаций шагов интерполяции и интегрирования брахистохроны')
    ax.set_xticks(np.log10(interp_steps_range))
    ax.set_xticklabels(interp_steps_range.astype(int))
    ax.set_yticks(np.log10(integration_steps_range[::10]))
    ax.set_yticklabels(integration_steps_range[::10].astype(int))
    ax.set_zticks(error_log_ticks)
    ax.set_zticklabels(error_log_labels)
    pdf.savefig()
    plt.show()


with PdfPages('figure4.pdf') as pdf:
    plt.figure(figsize=(10, 6))
    contour_levels = np.linspace(np.log10(errors.min()), np.log10(errors.max()), num=10)
    cp = plt.contour(np.log10(interp_steps_array), np.log10(integration_steps_array), np.log10(errors), levels=contour_levels)
    plt.clabel(cp, inline=True, fontsize=10)
    plt.xlabel('Шаги интерполяции')
    plt.ylabel('Шаги интегрирования')
    plt.title('Контурная диаграмма ошибки в зависимости\n от шагов интерполяции и интегрирования')
    plt.colorbar(cp, label='Error')
    pdf.savefig()
    plt.show()

min_error = np.min(errors)
min_error_index = np.unravel_index(errors.argmin(), errors.shape)
optimal_interp_steps = interp_steps_range[min_error_index[0]]
optimal_integration_steps = integration_steps_range[min_error_index[1]]

optimal_interp_steps, optimal_integration_steps, min_error


# In[26]:


from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cubic_spline_interpolation(t_points):
    t_values = np.linspace(0, T, t_points)
    cycloid_points = cycloid(t_values)

    cubic_spline_func = CubicSpline(cycloid_points[0], cycloid_points[1])

    return cubic_spline_func, cycloid_points[0]

def functional_value_cubic_spline(interp_points, integration_points):
    interp_func, x_values = cubic_spline_interpolation(interp_points)

    def integrand(x):
        y = interp_func(x)
        dy_dx = np.gradient(y, x)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sqrt((1 + dy_dx**2) / (2 * g * y))

    start_x = x_values[1] 
    end_x = x_values[-1]

    return composite_simpson(start_x, end_x, integration_points, integrand)

def optimization_objective(params):
    interp_points, integration_points = params
    approx_functional_value = functional_value_cubic_spline(int(interp_points), int(integration_points))
    error = np.abs(approx_functional_value - exact_functional_value)
    return error

initial_params = [50, 1000]  

param_bounds = [(10, 1000), (100, 10000)] 

result = minimize(optimization_objective, initial_params, bounds=param_bounds, method='L-BFGS-B')

result.fun, result.x 

optimal_interp_points = 10
cubic_spline_func, x_values = cubic_spline_interpolation(optimal_interp_points)

y_values_spline = cubic_spline_func(x_values)

exact_cycloid_points = cycloid(np.linspace(0, T, 1000))
def piecewise_linear_interpolation(t_points):
    t_values = np.linspace(0, T, t_points)
    cycloid_points = cycloid(t_values)

    linear_interp_func = interp1d(cycloid_points[0], cycloid_points[1], kind='linear', fill_value="extrapolate")

    return linear_interp_func, cycloid_points[0]

# Вычисляем оптимальное количество точек интерполяции для кусочно-линейной интерполяции
optimal_interp_points_linear = int(result.x[0])

# Вычисляем значения для кусочно-линейной интерполяции
linear_interp_func, x_values_linear = piecewise_linear_interpolation(optimal_interp_points_linear)
y_values_linear = linear_interp_func(x_values_linear)

# Добавляем график кусочно-линейной интерполяции на существующий график
with PdfPages('figure5.pdf') as pdf:
    plt.figure(figsize=(12, 8))
    plt.plot(x_values, y_values_spline, label='Кусочно-линейная интерполяция', color='blue')
    plt.plot(x_values_linear, y_values_linear, label='Кубический сплайн', linestyle='-.', color='green')
    plt.plot(exact_cycloid_points[0], exact_cycloid_points[1], label='Брахистохрона', linestyle='--', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Аппроксимация кубическими сплайнами и кусочно-линейной интерполяцией')
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.show()



interp_steps_range_cubic = np.array([10, 20, 50, 100, 200, 500, 1000])
integration_steps_range_cubic = np.logspace(-3, 0, 50) * 1000 

errors_cubic = np.zeros((len(interp_steps_range_cubic), len(integration_steps_range_cubic)))
interp_steps_array_cubic, integration_steps_array_cubic = np.meshgrid(interp_steps_range_cubic, integration_steps_range_cubic, indexing='ij')

for i, interp_points in enumerate(interp_steps_range_cubic):
    for j, integration_points in enumerate(integration_steps_range_cubic):
        approx_functional_value_cubic = functional_value_cubic_spline(interp_points, int(integration_points))
        errors_cubic[i, j] = np.abs(approx_functional_value_cubic - exact_functional_value)
with PdfPages('figure6.pdf') as pdf:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=135) 
    
    ax.plot_surface(np.log10(interp_steps_array_cubic), np.log10(integration_steps_array_cubic), np.log10(errors_cubic), cmap='viridis')
    
    ax.set_xlabel('Шаги интерполяции')
    ax.set_ylabel('Шаги интегрирования')
    ax.set_zlabel('L2 ошибка')
    ax.set_title('Анализ ошибки аппроксимации функционала для\n различных комбинаций шагов интерполяции и интегрирования брахистохроны \nс применением метода градиентного спуска и кубических сплайнов')
    pdf.savefig()
    plt.show()


# In[ ]:




