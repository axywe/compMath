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