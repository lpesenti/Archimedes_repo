import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from celluloid import Camera
import scipy.stats as stats
import math

fig, ax = plt.subplots()
x = np.arange(0, 50, 0.1)

camera = Camera(fig)

step = np.linspace(1, 50, 100, endpoint=False)

for i in step:
    t = ax.plot(x, np.exp(-i) * np.power(i, x) / factorial(x))
    u = ax.plot(x, stats.norm.pdf(x, i, math.sqrt(i)), color='black')
    ax.legend((t[0], u[0]), (rf'$\lambda$ = {round(i, 2)}', 'Normal distribution'))
    camera.snap()

animation = camera.animate()
animation.save('poisson_to_gauss.gif', writer='imagemagick')

plt.show()
