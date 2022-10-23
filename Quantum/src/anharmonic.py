"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import simps
from scipy import special
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


def display(fig, name):
    if(not DEBUG):
        fig.savefig(f"{name}.pgf", bbox_inches="tight", format="pgf")
        fig.savefig(f"{name}.png", bbox_inches="tight", format="png")
    if(DEBUG):
        plt.show()


def k_anharmonic(u, e):
    return 2 * (e - np.power(u, 2) / 2 - np.power(u, 4) / 4)


def k_harmonic(u, e):
    return 2 * (e - np.power(u, 2) / 2)


def k_free(u, e):
    return 2 * (e)


def iterate(i, e, v, v_prime, u, k, delx):
    v_prime[i + 1] = v_prime[i] - k(u - L / 2, e) * v[i] * delx
    v[i + 1] = v[i] + v_prime[i + 1] * delx


def shoot(e, v, v_prime, delx, n, k):
    for i in range(n):
        iterate(i, e, v, v_prime, i * delx, k, delx)


L = 10
n = 10000
delx = L / n
u = [delx * i - L / 2 for i in range(n + 1)]
v_prime = np.zeros(n + 1)
v = np.zeros(n + 1)
unit_v = np.zeros(n + 1)
# infinite potential well
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title(fr"Infinite potential well eigenstates")
axs.set_xlabel(r"Position $u$ [m]")
axs.set_ylabel(r"Wavefunction $v(u)$")
for k in range(1, 6):
    e = np.pi**2 / 2 / L / L * np.power(k, 2)
    v[0] = 0
    v_prime[0] = 1
    shoot(e, v, v_prime, delx, n, k_free)
    norm_v = simps(np.power(v, 2), u)
    unit_v = v / np.sqrt(norm_v)
    axs.plot(u, unit_v, label=f"$n={k}$")
fig.legend()
display(fig, f"../res/infwell")


fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title(fr"Harmonic oscillator eigenstates")
axs.set_xlabel(r"Position $u$ [m]")
axs.set_ylabel(r"Wavefunction $v(u)$")
eigenstate_num = 10
axs.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0, 1, eigenstate_num)))
fig2, axs2 = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs2.set_title(fr"Harmonic oscillator eigenstates")
axs2.set_xlabel(r"Position $u$ [m]")
axs2.set_ylabel(r"Wavefunction $v(u)$")
eigenstate_num_th = 4 * 2
axs2.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0, 1, eigenstate_num_th)))
tol = 1e-5
for q in range(eigenstate_num):
    emin = q
    emax = q + 1
    v[0] = 0
    unit_v[-1] = 1
    v_prime[0] = 1
    while (abs(unit_v[-1]) > tol):
        e = (emin + emax) / 2
        shoot(e, v, v_prime, delx, n, k_harmonic)
        norm_v = simps(np.power(v, 2), u)
        unit_v = v / np.sqrt(norm_v)
        if(abs(unit_v[-1]) < tol):
            if(q in {0, 1, 2, 3}):
                axs2.plot(u, unit_v, "*", alpha=0.1, label=fr"Numerics $\varepsilon={e:.1f}$")
                axs2.plot(u, [-((q % 2) * 2 - 1) * 1 / np.sqrt(np.power(2, q) * np.math.factorial(q) * np.sqrt(np.pi)) *
                              np.exp(-1 / 2 * el * el) * special.hermite(q)(el) for el in u], "--", label=fr"Theory $\varepsilon={e:.1f}$")
            axs.plot(u, unit_v, alpha=0.4, label=fr"$\varepsilon={e:.1f}$")
        if(((q % 2) * 2 - 1) * unit_v[-1] < 0):
            emin = e
        else:
            emax = e
axs.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
axs2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
display(fig, f"../res/harmonic")
display(fig2, f"../res/harmonic_th")

fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title(fr"Anharmonic oscillator eigenstates")
axs.set_xlabel(r"Position $u$ [m]")
axs.set_ylabel(r"Wavefunction $v(u)$")
eigenstate_num = 10
axs.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0, 1, eigenstate_num)))
tol = 1e-3
for q in range(eigenstate_num):
    emin = q - 1 + 9 * (q - 1) // 10 + 2 + max(q - 8, 0)
    emax = q + 9 * q // 10 + 2 + max(q - 8, 0)
    v[0] = 0
    unit_v[-1] = 1
    v_prime[0] = 1
    while (abs(unit_v[-1]) > tol):
        e = (emin + emax) / 2
        shoot(e, v, v_prime, delx, n, k_anharmonic)
        norm_v = simps(np.power(v, 2), u)
        unit_v = v / np.sqrt(norm_v)
        if(abs(unit_v[-1]) < tol):
            axs.plot(u, unit_v, alpha=0.4, label=fr"$\varepsilon={e:04.1f}$")
        if(((q % 2) * 2 - 1) * unit_v[-1] < 0):
            emin = e
        else:
            emax = e
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
display(fig, f"../res/anharmonic")
