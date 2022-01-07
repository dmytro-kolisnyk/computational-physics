"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List
import matplotlib
import random
# setup plotting
plt.style.use('seaborn-pastel')
# enables debug mode of the program (doesn't save graphs and disconnects
# from additional external dependencies, e.g TeX, FFMpeg, ...)
DEBUG = True
fontsize = 10
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'pgf.preamble': r'\usepackage{amsmath}',
    'font.family': 'serif',
    'text.usetex': not(DEBUG),  # enable TeX
    'text.latex.preamble': r'\usepackage{amsmath}',
    'pgf.rcfonts': False,
    'axes.grid': True,
    'grid.linestyle': '--',
    'legend.fontsize': fontsize,
    'axes.titlesize': fontsize,
    'axes.labelsize': fontsize,
    'figure.constrained_layout.use': True
})


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Calculates latex size of the figure to be plotted from pt size and number of subplots"""
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


@dataclass
class System:
    """
    Describes iterative map system
    """
    f: Callable[[float], float]
    state: float
    trajectory: List[float]

    def __init__(self, init_state: float, f: Callable[[float], float]):
        self.f = f
        self.state = init_state
        self.trajectory = [init_state]

    def update(self):
        """
        Iterate system state
        """
        self.state = self.f(self.state)
        self.trajectory.append(self.state)

    def evolve(self, iter_num):
        """
        Iterate system iter_num times
        """
        for i in range(iter_num):
            self.update()
        return self.trajectory


def logistic_map(r: float) -> Callable[[float], float]:
    """
    Define logistic map
    """
    return lambda x: 4 * r * x * (1 - x)


def d_logistic_map_num(r: float, x: float) -> Callable[[float], float]:
    """
    Calculate logistic map derivate at x
    """
    return  4 * r * (1 - 2 * x)


def exp_map(r: float) -> Callable[[float], float]:
    """
    Define Exp map
    """
    return lambda x: x * np.exp(4 * r * (1 - x))


def d_exp_map_num(r: float, x: float) -> Callable[[float], float]:
    """
    Calculate Exp map derivate at x
    """
    return  (1 - 4 * r * x) * np.exp(4 * r * (1 - x))


def sine_map(r: float) -> Callable[[float], float]:
    """
    Define Sine map
    """
    return lambda x: r * np.sin(np.pi * x)


def d_sine_map_num(r: float, x: float) -> Callable[[float], float]:
    """
    Calculate Sine map derivate at x
    """
    return r * np.pi * np.cos(np.pi * x)


def lyapunov_exp(x_0: float, r: float, func: Callable[[float], float],
                 d_func: Callable[[float], float]) -> float:
    """
    Calculate Lyapunov exponent using <f'(x_i)> formula
    """
    sys = System(x_0, func(r))
    sum = 0
    i = 1
    N = 1000
    sys.evolve(N)
    for i in range(N):
        sum += np.log(np.abs(d_func(r, sys.trajectory[i])))
    l_exp = sum / N
    return l_exp


init_state = [0.8, 0.8, 0.1, 0.09]
r = [0.2, 0.4, 0.73, 0.88]


def plot_map_evolution(r, init_state, map_name, func, color_map):
    """
    Plot iterations of a map
    """
    fig, axs = plt.subplots(2, 2, sharey=False, sharex=True, figsize=set_size(580, subplots=(2, 2)))
    n = 200
    x = np.arange(0, n + 1, 1)
    plot_num = 4
    cmap = plt.get_cmap(color_map)
    colors = [cmap(i) for i in np.linspace(0, 1, plot_num)]
    for i in range(plot_num):
        axs[list(map(lambda i:(i // 2, i % 2), [i]))[0]].plot(x, System(init_state[i], func(r[i])).evolve(n),
                                                              color=colors[i])
    xlabels = ["$n$" for i in range(4)]
    ylabels = ["$x(n)$" for i in range(4)]
    titles = [f"{map_name}: $r={r[i]}$, $x_0={init_state[i]}$" for i in range(4)]
    for i, ax in enumerate(axs.flat):
        ax.set(xlabel=xlabels[i], ylabel=ylabels[i], title=titles[i])
    if DEBUG:
        plt.show()
    else:
        plt.savefig(
            "../res/" + map_name +
            '.pgf',
            bbox_inches='tight',
            format='pgf')


def plot_error(r, init_state, init_state_shift, map_name, func, color_map):
    """
    Plot trajectory deviation for 2 different intial states of the system
    """
    fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
    n = 200
    x = np.arange(0, n + 1, 1)
    cmap = plt.get_cmap(color_map)
    colors = cmap(0.4)
    del_xn = np.array(System(init_state, func(r)).evolve(n)) - \
        np.array(System(init_state + init_state_shift, func(r)).evolve(n))
    axs.plot(x, np.log(np.abs(del_xn / del_xn[0])), color=colors)
    xlabels = "$n$"
    ylabels = r"$\ln{\left|\frac{\Delta x_n}{\Delta x_0}\right|}$"
    titles = rf"{map_name}: $r={r}$, $x_0={init_state}$, $\Delta x_0={init_state_shift}$"
    axs.set(xlabel=xlabels, ylabel=ylabels, title=titles)
    if DEBUG:
        plt.show()
    else:
        plt.savefig(
            "../res/" + map_name +
            '.pgf',
            bbox_inches='tight',
            format='pgf')


def plot_lambda(init_state, map_name, func, d_func, color_map):
    """
    Plot of Lyapunov exponent
    """
    fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
    n = 5000
    x = np.linspace(0.1, 1, n + 1)
    cmap = plt.get_cmap(color_map)
    colors = cmap(random.uniform(0, 1))
    axs.set_xticks(np.arange(0, 1.1, 0.1))
    axs.plot(x, [lyapunov_exp(init_state, el, func, d_func) for el in x], color=colors)
    xlabels = "$r$"
    ylabels = r"$\lambda$"
    titles = f"{map_name}: $x_0={init_state}$"
    axs.set(xlabel=xlabels, ylabel=ylabels, title=titles)
    if DEBUG:
        plt.show()
    else:
        plt.savefig(
            "../res/" + map_name +
            '.pgf',
            bbox_inches='tight',
            format='pgf')


# plot all map iterations
plot_map_evolution(r, init_state, "Logistic map", logistic_map, "tab10")
plot_map_evolution(r, init_state, "Exp map", exp_map, "tab10")
plot_map_evolution(r, init_state, "Sine map", sine_map, "tab10")
plot_error(0.92, 0.6, 1e-5, "Deviation of Logistic map trajectories", logistic_map, "tab10")
plot_lambda(0.3, "Lyapunov exponent - Logistic map", logistic_map, d_logistic_map_num, "tab10")
plot_lambda(0.3, "Lyapunov exponent - Exp map", exp_map, d_exp_map_num, "tab10")
plot_lambda(0.3, "Lyapunov exponent - Sine map", sine_map, d_sine_map_num, "tab10")
