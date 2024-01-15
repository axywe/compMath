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