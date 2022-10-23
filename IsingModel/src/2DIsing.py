"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
import copy
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# setup plotting

plt.style.use('seaborn-pastel')
dpi = 400
# enables debug mode of the program (disconnects from additional external dependencies, e.g LaTeX, FFMpeg, ...)
DEBUG = True
if(not DEBUG):
    matplotlib.use('Agg')
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


def E(A):
    """Calculate energy of spin lattice"""
    L = np.shape(A)[0]
    H = 0
    for i, j in product(range(L - 1), range(L - 1)):
        H += A[i, j] * A[i + 1, j] + A[i, j] * A[i, j + 1]
    for i, j in product(range(L - 1, L), range(L - 1)):
        H += A[i, j] * A[i, j + 1]
    for i, j in product(range(L - 1), range(L - 1, L)):
        H += A[i, j] * A[i + 1, j]
    return -H


def flip_random_spin(lattice):
    """Create a copy of the lattice with one random flipped spin"""
    L = np.shape(lattice)[0]
    res = copy.deepcopy(lattice)
    res[np.random.randint(0, L), np.random.randint(0, L)] *= -1
    return res


def flip_spin(lattice, i, j):
    """Flip spin at position [i,j] on the lattice"""
    lattice[i, j] *= -1


def r(delE, T):
    """Calculate probability factor for a transition in the Metropolis algorithm"""
    return np.exp(-(delE) / T)


def omega(delE, T):
    """Check if transition happens"""
    p = r(delE, T)
    if(p > 1):
        return True
    elif(p <= 0):
        return False
    else:
        if(np.random.uniform() < p):
            return True


def update(lattice, T):
    """Iterate lattice according to Metropolis algorithm"""
    L = np.shape(lattice)[0]
    delE = 0
    x, y = 0, 0
    s_nb = 0
    for i in range(np.shape(lattice)[0]**2):
        s_nb = 0
        x, y = np.random.randint(0, L), np.random.randint(0, L)
        if(y == 0):
            s_nb += lattice[x, y + 1]
        elif(y == L - 1):
            s_nb += lattice[x, y - 1]
        else:
            s_nb += lattice[x, y - 1] + lattice[x, y + 1]
        if(x == 0):
            s_nb += lattice[x + 1, y]
        elif(x == L - 1):
            s_nb += lattice[x - 1, y]
        else:
            s_nb += lattice[x - 1, y] + lattice[x + 1, y]
        delE = 2 * lattice[x, y] * s_nb
        if(omega(delE, T)):
            flip_spin(lattice, x, y)
    return lattice


def m(lattice):
    """Find lattice magnetization per site"""
    return np.abs(np.sum(lattice)) / (np.shape(lattice)[0]**2)


# set simulation parameters
L = 20
T_list = [0.1, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 10.0]
n0 = 6000
nmin = 2000
np.random.seed(2022)
shots = 20
seeds = [np.random.randint(100, 100000) for i in range(shots)]
E_avg_measured_list = []
m_avg_measured_list = []
E_err_measured_list = []
m_err_measured_list = []
for T, rand_seed in list(zip(T_list, seeds)):
    E_avg_list = []
    m_avg_list = []
    for rand_seed in seeds:
        np.random.seed(rand_seed)
        lattice = np.reshape([np.random.randint(0, 2) * 2 - 1 for i in range(L * L)], (L, L))
        E_history = [E(lattice)]
        m_history = [m(lattice)]
        E_tot = 0
        m_tot = 0
        for i in range(n0):
            lattice = update(lattice, T)
            m_history.append(m(lattice))
            E_history.append(E(lattice))
            if(i > (nmin - 1)):
                E_tot += E_history[-1]
                m_tot += m_history[-1]
        E_avg = E_tot / (n0 - nmin)
        m_avg = m_tot / (n0 - nmin)
        fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
        axs.set_title(f"System evolution according to Metropolis algorithm ($T'={T}$)")
        axs.set_xlabel("iteration number")
        axs.set_ylabel(r"$|\langle m\rangle|$")
        axs.plot(m_history, label=r"$\langle|\langle m\rangle|\rangle=" + f"{m_avg:.4f}" + "$")
        axs.legend()
        if(not DEBUG):
            fig.savefig(f"../res/m_{T*10}_{rand_seed}.pgf", bbox_inches="tight", format="pgf")
            fig.savefig(f"../res/m_{T*10}_{rand_seed}.png", bbox_inches="tight", format="png")
        else:
            plt.show()
        fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
        axs.set_title(f"System evolution according to Metropolis algorithm ($T'={T}$)")
        axs.set_xlabel("iteration number")
        axs.set_ylabel(r"$E'$")
        axs.plot(E_history, label=rf"$\langle E'\rangle=" + f"{E_avg:.4f}" + "$")
        axs.legend()
        if(not DEBUG):
            fig.savefig(f"../res/e_{T*10}_{rand_seed}.pgf", bbox_inches="tight", format="pgf")
            fig.savefig(f"../res/e_{T*10}_{rand_seed}.png", bbox_inches="tight", format="png")
        else:
            plt.show()
        fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
        axs.set_title(f"Final spin configuration ($T'={T}$)")

        axs = sns.heatmap(
            lattice, square=True, linewidth=0.2,
            xticklabels=[i for i in range(1, L + 1)],
            yticklabels=[i for i in range(1, L + 1)],
            vmin=-1,
            vmax=1)
        if(not DEBUG):
            fig.savefig(f"../res/2d_lattice_{T*10}_{rand_seed}.pgf", bbox_inches="tight", format="pgf")
            fig.savefig(f"../res/2d_lattice_{T*10}_{rand_seed}.png", bbox_inches="tight", format="png")
        else:
            plt.show()
        print("T:", T, "| E_avg:", E_avg, "| m_avg:", m_avg)
        E_avg_list.append(E_avg)
        m_avg_list.append(m_avg)
    E_avg_measured_list.append(np.mean(np.array(E_avg_list)))
    m_avg_measured_list.append(np.mean(np.array(m_avg_list)))
    E_err_measured_list.append(np.std(E_avg_list, ddof=1) / np.sqrt(np.size(E_avg_list)))
    m_err_measured_list.append(np.std(m_avg_list, ddof=1) / np.sqrt(np.size(m_avg_list)))

fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title(r"Magnetization per site dependence on temperature $|\langle m\rangle|(T)$")
axs.errorbar(T_list, m_avg_measured_list, yerr=m_err_measured_list, capsize=4, label="Simulation results")
resolution = 100
T_space = np.linspace(min(T_list), max(T_list), resolution)
axs.plot(T_space, np.power(np.fmax((1 - np.power(np.sinh(2 / T_space), -4)),
                                   np.zeros(resolution)), 1 / 8), label="Onsager's solution")
axs.legend()
if(not DEBUG):
    fig.savefig(f"../res/magnetization.pgf", bbox_inches="tight", format="pgf")
    fig.savefig(f"../res/magnetization.png", bbox_inches="tight", format="png")
else:
    plt.show()
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title(r"Total energy dependence on temperature $E'(T)$")
axs.errorbar(T_list, E_avg_measured_list, yerr=E_err_measured_list, capsize=4, label="Simulation results")
axs.legend()
if(not DEBUG):
    fig.savefig(f"../res/energy.pgf", bbox_inches="tight", format="pgf")
    fig.savefig(f"../res/energy.png", bbox_inches="tight", format="png")
else:
    plt.show()
