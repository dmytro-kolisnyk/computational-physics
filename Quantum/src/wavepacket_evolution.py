"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.fft import fft, ifft, fftfreq
from matplotlib.animation import FuncAnimation
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


def display(fig, name):
    if(not DEBUG):
        fig.savefig(f"{name}.pgf", bbox_inches="tight", format="pgf")
        fig.savefig(f"{name}.png", bbox_inches="tight", format="png")
    if(DEBUG):
        plt.show()


def init_v(u):
    return np.power(np.pi, -1 / 4) * np.exp(-np.power(u, 2) / 2)


def V_free(u):
    return np.zeros_like(u)


def V_barrier(u, V0, a, b):
    V = np.zeros_like(u)
    for i, el in enumerate(u):
        if(el >= a and el <= b):
            V[i] = V0
    return V


def V_harmonic(u, a):
    return a * np.power(u, 2)


def T_class(u_tilde):
    return np.power(u_tilde, 2) / 2


def iterate(psi, T_tilde, V, dt):
    return np.exp(-1j * V * dt / 2) * ifft(np.exp(-1j * T_tilde * dt) * fft(np.exp(-1j * V * dt / 2) * psi))


L = 100
n = 500
delx = L / (n)
u = np.array([delx * i - L / 2 for i in range(n)])
u_tilde = fftfreq(n, delx) * (2 * np.pi)
# free evolution of a gaussian
dt = 0.1
v = init_v(u)
norm_v = np.real(np.sum(v.conj() * v) * delx)
print("Initial norm:", norm_v, "[free evolution, initial momentum 0]")
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
line, = axs.plot([], [], lw=2)


def init():
    axs.set_xlim([-L // 2, L // 2])
    axs.set_ylim([0, 0.6])
    axs.set_title(r"Free gaussian evolution")
    axs.set_xlabel(r"Position $u$ [m]")
    axs.set_ylabel(r"Probability density $|v(u)|^2$")
    line.set_data([], [])
    return line,


def animate(i):
    global v, u
    v = iterate(v, T_class(u_tilde), V_free(u), dt)
    line.set_data(u, np.real(v * v.conj()))
    return line,


anim = FuncAnimation(fig, animate, init_func=init, frames=100, interval=20, blit=True)
if(DEBUG):
    plt.show()
else:
    anim.save('../anims/free.mp4', writer='ffmpeg', fps=30)
norm_v = np.real(np.sum(v.conj() * v) * delx)
print("  Final norm:", norm_v, "[free evolution, initial momentum 0]")

# Evolution of a gaussian with some initial momentum near a potential barrier
dt = 0.01
k0 = 10
a = 3
b = 3.1
V0 = 10


v = init_v(u) * np.exp(1j * k0 * u)
norm_v = np.real(np.sum(v.conj() * v) * delx)
print("Initial norm:", norm_v, f"[evolution, initial momentum {k0}]")
v_tilde = fft(v)
print("Initial momentum exp val:", np.real(np.sum(v_tilde.conj() * u_tilde * v_tilde) / len(v) * delx))

fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
line, = axs.plot([], [], lw=2)
line_pot, = axs.plot([], [], lw=2)


def init():
    axs.set_xlim([-L // 2, L // 2])
    axs.set_ylim([0, 1])
    axs.set_title(r"Gaussian with initial momentum hitting a potential barrier")
    axs.set_xlabel(r"Position $u$ [m]")
    axs.set_ylabel(r"Probability density $|v(u)|^2$ (blue)" + "\n" + r"Potential $V/100$ (green)")
    line.set_data([], [])
    line_pot.set_data([], [])
    return (line, line_pot,)


T = T_class(u_tilde)
V = V_barrier(u, V0, a, b)


def animate(i):
    global v, u, T, V
    v = iterate(v, T, V, dt)
    line.set_data(u, np.real(v.conj() * v))
    line_pot.set_data([u], [V / 100])
    return (line, line_pot)


anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
if(DEBUG):
    plt.show()
else:
    anim.save(f'../anims/withpbarrier_{a}_{b}_{V0}_{k0}.mp4', writer='ffmpeg', fps=30)
norm_v = np.real(np.sum(v.conj() * v) * delx)
print("  Final norm:", norm_v, f"[evolution, initial momentum {k0}]")

# Evolution of a gaussian with some initial momentum within a harmonic potential
dt = 0.1
k0 = 10  # 100
u0 = 5
v = init_v(u - np.ones_like(u) * u0) * np.exp(1j * k0 * (u - np.ones_like(u) * u0))
norm_v = np.real(np.sum(v.conj() * v) * delx)
print("Initial norm:", norm_v, f"[evolution within harmonic oscillator, initial momentum {k0}]")

v_tilde = fft(v)
print("Initial position exp val:", np.real(np.sum(v.conj() * u * v) * delx))
print("Initial momentum exp val:", np.real(np.sum(v_tilde.conj() * u_tilde * v_tilde) / len(v) * delx))

fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
line, = axs.plot([], [], lw=2)
line_pot, = axs.plot([], [], lw=2)


def init():
    axs.set_xlim([-L // 2, L // 2])
    axs.set_ylim([0, 2])
    axs.set_title(r"Gaussian with initial momentum within a harmonic potential")
    axs.set_xlabel(r"Position $u$ [m]")
    axs.set_ylabel(r"Probability density $|v(u)|^2$ (blue)" + "\n" + r"Potential $V$ (green)")
    line.set_data([], [])
    line_pot.set_data([], [])
    return (line, line_pot,)


T = T_class(u_tilde)
a = 0.5
V = V_harmonic(u, a)

u_exp = [np.sum(v.conj() * u * v) * delx]
v_tilde = fft(v)
k_exp = [np.sum(v_tilde.conj() * u_tilde * v_tilde) / len(v) * delx]


def animate(i):
    global v, u, T, V, u_exp, k_exp
    v = iterate(v, T, V, dt)
    line.set_data(u, np.real(v * v.conj()))
    line_pot.set_data([u], [V])
    u_exp.append(np.sum(v.conj() * u * v) * delx)
    v_tilde = fft(v)
    k_exp.append(np.sum(v_tilde.conj() * u_tilde * v_tilde) / len(v) * delx)
    return (line, line_pot,)


anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)
if(DEBUG):
    plt.show()
else:
    anim.save(f'../anims/withpharmonic_{a}_{k0}_{u0}.mp4', writer='ffmpeg', fps=30)
norm_v = np.real(np.sum(v.conj() * v) * delx)
print("  Final norm:", norm_v, f"[evolution within harmonic oscillator, initial momentum {k0}]")
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title(r"Evolution of the position expectation value")
axs.set_xlabel(rf"Time $t$ [${dt}\cdot$s]")
axs.set_ylabel(r"Position expectation value $\langle u(t) \rangle$")
axs.plot(np.real(u_exp))
display(fig, f"../res/exp_values_harmonic_u_{a}_{k0}_{u0}")

fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title(r"Evolution of the momentum expectation value")
axs.set_xlabel(rf"Time $t$ [${dt}\cdot$s]")
axs.set_ylabel(r"Momentum expectation value $\langle k(t) \rangle$")
axs.plot(np.real(k_exp))
display(fig, f"../res/exp_values_harmonic_k_{a}_{k0}_{u0}")
