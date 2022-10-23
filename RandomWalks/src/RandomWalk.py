"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# setup plotting
plt.style.use('seaborn-pastel')
dpi = 400
# enables debug mode of the program (disconnects from additional external dependencies, e.g FFMpeg, ...)
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


def n_random_steps(p, n):
    """
    Perform n random steps
    Return displacement
    """
    q = 1 - p
    pos = 0
    for i in range(n):
        pos += rng.choice([-1, 1], p=[p, q])
    return pos


def random_walk_to_barrier(p, x0, max_n):
    """
    Count number of steps until the barrier is reached
    Return -1 if the barrier was not reached
    """
    q = 1 - p
    pos = 0
    i = 0
    while (x0 != pos) and (i < max_n):
        pos += rng.choice([-1, 1], p=[p, q])
        i += 1
    if(x0 == pos):
        return i
    return -1


rng = np.random.default_rng(2022)
# displacement distribution analysis
samples_num = 100
p = 0.5
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Distribution of final displacements after $n$ steps")
axs.set_xlabel("Final displacement")
axs.set_ylabel("PDF")
means = []
variances = []
n_range = range(0, 4011, 1000)
for n in n_range:
    final_displacements = []
    for j in range(samples_num):
        final_displacements.append(n_random_steps(p, n))
    means.append(np.average(final_displacements))
    variances.append(np.var(final_displacements))
    if n != 0:
        axs.hist(final_displacements, bins=10, density=True, alpha=0.5, label=f"$n={n}$")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/DisplacementDistribution.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/DisplacementDistribution.png", bbox_inches="tight", format="png")
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("First and second moments of final displacement distribution")
axs.set_xlabel("$n$")
axs.plot(n_range, means, label=r"$\langle x \rangle$")
axs.plot(n_range, variances, label=r"$\sigma^2$")
axs.plot([0, n_range[-1]], [0, n_range[-1]], label=r"$\sigma^2=n$ line")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/MomentsOfDisplacementDistribution.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/MomentsOfDisplacementDistribution.png", bbox_inches="tight", format="png")
# random walk with barrier analysis (may take a few minutes)
ns = []
x0 = 10
w = 1000
for i in range(w):
    n = random_walk_to_barrier(0.5, x0, 10000)
    if(n != -1):
        ns.append(np.log10(n))
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title(f"Distribution of the number of iterations before $x_0={x0}$ displacement is reached")
axs.set_ylabel(f"Number of occasions in a random\nsample with {w} walkers")
axs.set_xlabel("Number of iterations (in powers of 10, i.e. log10 scaled number of iterations)")
axs.hist(ns, bins=20, alpha=1, log=True, label=f"$n={n}$")
if(not DEBUG):
    plt.savefig("../res/StepsToBarrierStart10.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/StepsToBarrierStart10.png", bbox_inches="tight", format="png")
ns = []
x0 = 32
w = 1000
for i in range(w):
    n = random_walk_to_barrier(0.5, x0, 10000)
    if(n != -1):
        ns.append(np.log10(n))
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title(f"Distribution of the number of iterations before $x_0={x0}$ displacement is reached")
axs.set_ylabel(f"Number of occasions in a random\nsample with {w} walkers")
axs.set_xlabel("Number of iterations (in powers of 10, i.e. log10 scaled number of iterations)")
axs.hist(ns, bins=20, alpha=1, log=True, label=f"$n={n}$")
if(not DEBUG):
    plt.savefig("../res/StepsToBarrierStart32.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/StepsToBarrierStart32.png", bbox_inches="tight", format="png")
if DEBUG:
    plt.show()
