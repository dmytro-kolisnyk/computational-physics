"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

Parameters of line fits y = ax + b (as [a, b]):
Expected slopes: -1/N = -0.50; -1; -0.5
Rect error fit: [-0.54432972  1.3798948 ] MC sample variance fit: [-1.02648934 -2.01950888] MC error fit: [-0.79461152  0.31352242]
Expected slopes: -1/N = -0.33; -1; -0.5
Rect error fit: [-0.37491861  1.62073754] MC sample variance fit: [-0.98344892 -0.62038351] MC error fit: [-0.53033197 -0.80662506]
Expected slopes: -1/N = -0.25; -1; -0.5
Rect error fit: [-0.32106658  2.0626621 ] MC sample variance fit: [-1.03481899  1.04541784] MC error fit: [-0.7108829   1.48063285]
Expected slopes: -1/N = -0.20; -1; -0.5
Rect error fit: [-0.40141037  2.89702633] MC sample variance fit: [-1.0875491   2.11902384] MC error fit: [-0.38575383 -0.90531009]
Expected slopes: -1/N = -0.17; -1; -0.5
Rect error fit: [-0.57093343  4.43820292] MC sample variance fit: [-1.01359783  2.62634459] MC error fit: [-0.4023833  -0.73742545]

"""
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
# setup plotting
plt.style.use('seaborn-pastel')
dpi = 400
# enables debug mode of the program (disconnects from additional external dependencies, e.g LaTeX, FFMpeg, ...)
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


def sphere_volume_rectangular(N, n_p):
    """
    Calculates the volume of the unit N-dimensional hypersphere using
    rectangular integration method with n_p integration points
    """
    n = int(np.ceil(np.power(n_p, 1 / N)))
    x = np.linspace(0, 1, n)
    count = 0
    for i, r in enumerate(product(*[x for i in range(N)])):
        if(1 - np.linalg.norm(r)) > 0:
            count += 1
    return np.power(2, N) * count / np.power(n, N)


def sphere_volume_Monte_Carlo(N, n_p):
    """
    Calculates the volume of the unit N-dimensional hypersphere using
    Monte Carlo integration method with n_p integration points
    """
    n = int(np.ceil(np.power(n_p, 1 / N)))
    count = 0
    for i, r in enumerate([np.random.uniform(size=N) for i in range(int(np.power(n, N)))]):
        if(1 - np.linalg.norm(r)) > 0:
            count += 1
    return np.power(2, N) * count / np.power(n, N)


def generate_volumes(n=8, V_0=1, V_1=2):
    """
    Generates the volumes of the unit N=0,...,n-dimensional hypersphere using
    a known recursion relation
    """
    volumes = np.zeros(n)
    volumes[0] = V_0
    volumes[1] = V_1
    for i in range(2, n):
        volumes[i] = 2 * np.pi / i * volumes[i - 2]
    return volumes


# calculate exact volumes of hyperspheres
V = generate_volumes()
# prepare plots
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Error of rectangular integration of the unit hypersphere")
axs.set_xlabel(r"$n_p$")
axs.set_ylabel(r"$|V_\mathrm{rect}-V|$")
figw, axsw = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axsw.set_title("Error of Monte Carlo integration of the unit hypersphere")
axsw.set_xlabel(r"$n_p$")
axsw.set_ylabel(r"$|V_\mathrm{MC}-V|$")
figmc, axsmc = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axsmc.set_title("Sample variance of Monte Carlo unit hypersphere integration ensemble of results")
axsmc.set_xlabel(r"$n_p$")
axsmc.set_ylabel(r"$\tilde{\sigma}^2$")
# allocate necessary arrays
x = []
y = []
z = []
w = []
# generate seeds for Monte Carlo ensembles
mc_samples_num = 20
np.random.seed(2022)
seeds = [np.random.randint(2022, 2000022) for i in range(mc_samples_num)]
V_Monte_Carlo = []
V_Monte_Carlo_sq = []
last_points_num_to_fit = 8
q = 10
colors = plt.cm.rainbow(np.linspace(0, 1, q))
# find rectangular integration errors (y), sample variance of Monte Carlo
# integration results (z), Monte Carlo integration error (w); for different
# dimensions N and number of integration points (x)
for N in range(2, 7):
    x = []
    y = []
    z = []
    w = []
    for n in range(4, 11):
        V_Monte_Carlo = []
        V_Monte_Carlo_sq = []
        x.append(np.power(np.ceil(np.power(np.power(10, n / 2), 1 / N)), N))
        y.append(np.abs(sphere_volume_rectangular(N, x[-1]) - V[N]))
        for seed in seeds:
            np.random.seed(seed)
            V_Monte_Carlo.append(sphere_volume_Monte_Carlo(N, x[-1]))
            V_Monte_Carlo_sq.append(np.power(V_Monte_Carlo[-1], 2))
        z.append((np.average(V_Monte_Carlo_sq) - np.power(np.average(V_Monte_Carlo), 2)) / mc_samples_num)
        w.append(np.abs(np.average(V_Monte_Carlo) - V[N]))
    axs.loglog(x, y, label=rf"$N$ = {N}", alpha=0.5, color=colors[N])
    x_log = np.log(np.array(x)[-last_points_num_to_fit:])
    y_log = np.log(np.array(y)[-last_points_num_to_fit:])
    z_log = np.log(np.array(z)[-last_points_num_to_fit:])
    w_log = np.log(np.array(w)[-last_points_num_to_fit:])
    coefs_axs = np.polyfit(x_log, y_log, 1)
    coefs_axsmc = np.polyfit(x_log, z_log, 1)
    coefs_axsw = np.polyfit(x_log, w_log, 1)
    print(f"Expected slopes: -1/N = {-1/N:.2f};", "-1;", "-0.5", "\nRect error fit:",
          coefs_axs, "MC sample variance fit:", coefs_axsmc, "MC error fit:", coefs_axsw)
    axs.loglog(x[-last_points_num_to_fit:], np.exp(np.poly1d(coefs_axs)(x_log)), "--", color=colors[N])
    axsmc.loglog(x, z, label=rf"$N$ = {N}", alpha=0.5, color=colors[N])
    axsmc.loglog(x[-last_points_num_to_fit:], np.exp(np.poly1d(coefs_axsmc)(x_log)), "--", color=colors[N])
    axsw.loglog(x, w, label=rf"$N$ = {N}", alpha=0.5, color=colors[N])
    axsw.loglog(x[-last_points_num_to_fit:], np.exp(np.poly1d(coefs_axsw)(x_log)), "--", color=colors[N])

axs.legend()
axsmc.legend()
axsw.legend()
if(not DEBUG):
    fig.savefig("../res/rectloglog.pgf", bbox_inches="tight", format="pgf")
    fig.savefig("../res/rectloglog.png", bbox_inches="tight", format="png")
    figmc.savefig("../res/mcloglog.pgf", bbox_inches="tight", format="pgf")
    figmc.savefig("../res/mcloglog.png", bbox_inches="tight", format="png")
    figw.savefig("../res/mcerrloglog.pgf", bbox_inches="tight", format="pgf")
    figw.savefig("../res/mcerrloglog.png", bbox_inches="tight", format="png")
else:
    plt.show()
