"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
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


def update(u, beta):
    """updates wave lattice via Lax-Wendroff scheme"""
    u_new = np.empty_like(u)
    u_new[0] = 0
    u_new[len(u) - 1] = 0
    for i in range(1, len(u) - 1):
        u_new[i] = u[i] - beta / 4 * (u[i + 1]**2 - u[i - 1]**2) + beta**2 / 8 * ((u[i + 1] - u[i])
                                                                                  * (u[i + 1]**2 - u[i]**2) - (u[i] - u[i - 1]) * (u[i]**2 - u[i - 1]**2))
    return u_new



def animate(i, traj):
    """Updates animation frame"""
    global line
    line.set_ydata(traj[i])  # update the data.
    """
    time = (i/p)*k
    plt.legend(["$t=%4.2f$" % time])
    """
    return line,

def simulate_wave(u, k, h, epsilon, N, DEBUG, p, frames_num, interval):
    """
    Runs simulation of wave using method specified in update function
    """
    global line
    # calculate CFL number
    beta = epsilon / (h / k)
    u0 = u
    # setup plot for graphing
    fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
    axs.set_title("Wave evolution with $k="+str(k)+"$ [s], $h="+str(h)+r"$ [m], $\beta="+str(beta)+"$")
    axs.set_xlabel("$x$ [m]")
    axs.set_ylabel("$u(x)$")
    x = h * np.arange(0, N)
    # plot colors
    colors = [
        "coral",
        "olivedrab",
        "lime",
        "aquamarine",
        "deepskyblue",
        "lawngreen",
        "greenyellow",
        "limegreen",
        "forestgreen",
        "green",
        "darkgreen"]
    line, = axs.plot(x, u, colors[0], label="$t=0$ [s]")
    # save trajectory of the system
    traj = []
    for i in range(frames_num):
        # append all frames required for animation to trajectory
        for j in range(p):
            traj.append(u)
        # plot waves with large enough time intervals
        if((i % interval == 0) and (i != 0)):
            time = i*k
            axs.plot(x, u, colors[(i // interval)], label=(f"$t={time:.2f}") + "$ [s]")
        # update system 1 iteration forward
        u = update(u, beta)
    # add legend
    plt.legend(ncol=2)
    if(DEBUG):
        # show obtained plot
        plt.show()
    else:
        # save plot and complementary animation
        plt.savefig("../res/SampleWaves_"+str(int(beta*100))+".pgf", bbox_inches="tight", format="pgf")
        plt.savefig("../res/SampleWaves_"+str(int(beta*100))+".png", bbox_inches="tight", format="png")
        fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
        axs.set_title("Wave evolution with $k="+str(k)+"$ [s], $h="+str(h)+r"$ [m], $\beta="+str(beta)+"$")
        axs.set_xlabel("$x$ [m]")
        axs.set_ylabel("$u(x)$")
        axs.set_ylim(min(traj[-1])-0.4,max(traj[-1])+0.4)
        line, = axs.plot(x, u0, colors[0], label="$t=0$")
        ani = animation.FuncAnimation(fig, animate, fargs=(traj,), blit=True, frames=frames_num * p)
        writer = animation.FFMpegWriter(fps=p, metadata=dict(artist='Dmytro Kolisnyk'), bitrate=1800)
        ani.save("../res/wave_"+str(int(beta*100))+".mp4", writer=writer, dpi=dpi)

# set lattice size
N = 101
# intitalize arrays, storing system state
u0 = np.empty(N)
u = np.zeros(N)
# set parameters of simulation
k = 0.1
h = 0.05
p = 5
c = 1
# set initial value of the wave
u0 = np.array([3 * np.sin(3.2 * i * h) for i in range(N)])
u = np.copy(u0)
u[0] = 0
u[N - 1] = 0
# indirectly choose CFL number and run the simulation for various CFL numbers 
epsilon = 0.01
simulate_wave(u, k, h, epsilon, N, DEBUG, p, 200, 40)
epsilon = 0.09
simulate_wave(u, k, h, epsilon, N, DEBUG, p, 50, 10)
epsilon = 0.18
simulate_wave(u, k, h, epsilon, N, DEBUG, p, 10, 2)
epsilon = 0.27
simulate_wave(u, k, h, epsilon, N, DEBUG, p, 5, 1)
epsilon = 0.36
simulate_wave(u, k, h, epsilon, N, DEBUG, p, 5, 1)
epsilon = 0.45
simulate_wave(u, k, h, epsilon, N, DEBUG, p, 5, 1)
epsilon = 0.54
simulate_wave(u, k, h, epsilon, N, DEBUG, p, 5, 1)
epsilon = 0.63
simulate_wave(u, k, h, epsilon, N, DEBUG, p, 5, 1)
epsilon = 0.72
simulate_wave(u, k, h, epsilon, N, DEBUG, p, 5, 1)