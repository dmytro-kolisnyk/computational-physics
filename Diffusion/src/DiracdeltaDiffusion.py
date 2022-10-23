"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import solve_banded
from scipy.integrate import simps
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


def B(r, P):
    """Function B of vector P needed for Crank-Nicolson scheme"""
    res = np.copy(P)
    res[0] = 0
    res[-1] = 0
    for i in range(1, len(P) - 1):
        res[i] = (2 - 2 * r) * P[i] + r * (P[i + 1] + P[i - 1])
    return res


def evolve(A, P, r):
    """Solve equation AP(n+1)=B(P(n)) for P(n+1) need for Crank-Nicolson scheme"""
    return solve_banded((1, 1), A, B(r, P))

# setup simulation parameters
nx = 1000
delt = 1e-9
delx = 1e-6
D = (delx**2) / delt
r = D * delt / (delx**2)
x = np.array([i * delx for i in range(nx)])
Pcurr = np.zeros(nx)
Pnext = np.zeros(nx)
A = np.array([[0] + [-r for i in range(nx - 1)],
              [2 * r + 2 for i in range(nx)],
              [-r for i in range(nx - 1)] + [0]])
# start with delta distribution in the middle
Pcurr[nx // 2] = 1
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Dirac delta density $P(x,t=0)$ plot")
axs.set_xlabel(r"$x_i/\Delta x$")
axs.set_ylabel("$P(x)$")
axs.plot(Pcurr, label=r"$t/\Delta t=0$, " + r"$P_\mathrm{tot}=" + f"{simps(Pcurr):.2f}" + "$")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/dirac.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/dirac.png", bbox_inches="tight", format="png")
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Density $P(x,t)$ plot")
axs.set_xlabel(r"$x_i/\Delta x$")
axs.set_ylabel("$P(x)$")
iter_num = 20000
mean = []
variance = []
mean.append(sum([i*p for i,p in enumerate(Pcurr/simps(Pcurr))]))
variance.append(sum([i*i*p for i,p in enumerate(Pcurr/simps(Pcurr))])-mean[-1]**2)
for i in range(iter_num):
    Pnext = evolve(A, Pcurr, r)
    Pcurr = Pnext
    mean.append(sum([i*p for i,p in enumerate(Pcurr/simps(Pcurr))]))
    variance.append(sum([i*i*p for i,p in enumerate(Pcurr/simps(Pcurr))])-mean[-1]**2)
    if((i + 1) % (iter_num // 5) == 0):
        axs.plot(Pcurr, label=rf"$t/\Delta t={(i+1)}$, " + r"$P_\mathrm{tot}=" + f"{simps(Pcurr):.2f}" + "$")
        # print(i+1,variance[i],2*r*(i+1)) # compare empirical variance to a formula
plt.legend()
if(not DEBUG):
    plt.savefig("../res/diracd.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/diracd.png", bbox_inches="tight", format="png")
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Mean and variance of density distribution")
axs.set_xlabel(r"$t_i/\Delta t$")
axs.plot(mean, label=r"$\langle x \rangle$")
axs.plot(variance, label=r"$\sigma^2$")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/mom.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/mom.png", bbox_inches="tight", format="png")
# if(DEBUG):
#     plt.show()
# continue with delta distribution closer to the boundary
Pcurr = np.zeros(nx)
Pcurr[nx // 100] = 1
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Dirac delta density $P(x,t=0)$ plot")
axs.set_xlabel(r"$x_i/\Delta x$")
axs.set_ylabel("$P(x)$")
axs.plot(Pcurr, label=r"$t/\Delta t=0$, " + r"$P_\mathrm{tot}=" + f"{simps(Pcurr):.2f}" + "$")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/ndirac.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/ndirac.png", bbox_inches="tight", format="png")
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Density $P(x,t)$ plot")
axs.set_xlabel(r"$x_i/\Delta x$")
axs.set_ylabel("$P(x)$")
iter_num = 10000
mean = []
variance = []
mean.append(sum([i*p for i,p in enumerate(Pcurr/simps(Pcurr))]))
variance.append(sum([i*i*p for i,p in enumerate(Pcurr/simps(Pcurr))])-mean[-1]**2)
diff = []
for i in range(iter_num):
    Pnext = evolve(A, Pcurr, r)
    diff.append(simps(Pcurr) - simps(Pnext))
    Pcurr = Pnext
    mean.append(sum([i*p for i,p in enumerate(Pcurr/simps(Pcurr))]))
    variance.append(sum([i*i*p for i,p in enumerate(Pcurr/simps(Pcurr))])-mean[-1]**2)
    if((i + 1) % (iter_num // 5) == 0):
        axs.plot(Pcurr, label=rf"$t/\Delta t={i+1}$, " + r"$P_\mathrm{tot}=" + f"{simps(Pcurr):.2f}" + "$")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/ndiracd.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/ndiracd.png", bbox_inches="tight", format="png")
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Mean and variance through boundaries of density distribution")
axs.set_xlabel(r"$t_i/\Delta t$")
axs.plot(mean, label=r"$\langle x \rangle$")
axs.plot(variance, label=r"$\sigma^2$")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/nmom.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/nmom.png", bbox_inches="tight", format="png")
# analyse outgoing flux through the boundaries
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Flux through boundaries of density distribution")
axs.set_xlabel(r"$t_i/\Delta t$")
axs.plot(diff, label=r"$\Delta P_\mathrm{tot}$")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/flux.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/flux.png", bbox_inches="tight", format="png")
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Flux through boundaries of density distribution (log10ylog10x scale)")
axs.set_xlabel(r"$t_i/\Delta t$")
axs.set_ylabel("$f(t)$")
axs.loglog(diff, label=r"$\Delta P_\mathrm{tot}$")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/llnmom.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/llnmom.png", bbox_inches="tight", format="png")
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Flux through boundaries of density distribution (log10y scale)")
axs.set_xlabel(r"$t_i/\Delta t$")
axs.set_ylabel("$f(t)$")
axs.set_yscale("log", base=10)
axs.plot(diff, label=r"$\Delta P_\mathrm{tot}$")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/lnmom.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/lnmom.png", bbox_inches="tight", format="png")
if DEBUG:
    plt.show()
