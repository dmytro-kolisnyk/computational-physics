"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
from __future__ import annotations
from math import sqrt
from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pylab as plt
"""
For exporting graphs as pgf files

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': True,
})
"""


@dataclass
class Particle:
    """
    Represents a particle with some mass, position, velocity, acceleration
    """
    m: float
    r: np.ndarray
    v: np.ndarray
    a: np.ndarray

    def update(self, upd_a: np.ndarray, dt: float):
        """Euler scheme"""
        self.a = upd_a
        self.v = self.v + self.a * dt
        self.r = self.r + self.v * dt

    def __init__(self, m: float, r: np.ndarray, v: np.ndarray, a: np.ndarray):
        self.m = m
        self.r = r
        self.v = v
        self.a = a


@dataclass
class System:
    """
    Represents simulated system with all the parameters needed
    Part. case: timestep dt, G and list of all particles define the system
    """
    dt: float
    G: float
    p_list: List[Particle]

    def F_over_m(self, p_id: float) -> np.ndarray:
        """
        Calculates acceleration of particle with a specific p_id
        (Force acting on a spec. particle is a sum of grav. pulls, 
        created by all other particles in the system)
        """
        s = sum([-self.G * other_p.m / (np.linalg.norm(self.p_list[p_id].r - other_p.r)**3)
                 * (self.p_list[p_id].r - other_p.r) for i, other_p in enumerate(self.p_list) if i != p_id])
        return s

    def calc_E_pot(self, p_id: float):
        """
        Calculates energy of interaction of a specific particle with all other
        particles in the system
        """
        return -self.G * self.p_list[p_id].m * sum([self.p_list[i].m / np.linalg.norm(
            self.p_list[p_id].r - self.p_list[i].r) for i, other_p in enumerate(self.p_list) if i != p_id])

    def update(self):
        """
        Updates state of the system by updating state of each particle
        """
        for i, p in enumerate(self.p_list):
            self.p_list[i].update(self.F_over_m(i), self.dt)

    def __init__(self, dt: float, G: float, p_list: List[Particle]):
        self.dt = dt
        self.G = G
        self.p_list = p_list


@dataclass
class ParticleHistory:
    """
    Represents data of interest of a particle which should be saved 
    during the simulation
    """
    m: float
    r: list
    v: list
    E_kin: list
    E_pot: list

    def __init__(self, p_id: float, sys: System):
        p = sys.p_list[p_id]
        self.m = p.m
        self.r = [p.r]
        self.v = [p.v]
        self.E_kin = [p.m * (np.linalg.norm(p.v)**2) / 2]
        self.E_pot = [sys.calc_E_pot(p_id)]

    def record(self, p_id: float, sys: System):
        """
        Records current state of the particle
        """
        r = sys.p_list[p_id].r
        v = sys.p_list[p_id].v
        self.r.append(r)
        self.v.append(v)
        E_k = self.m * (np.linalg.norm(v)**2) / 2
        E_p = sys.calc_E_pot(p_id)
        self.E_kin.append(E_k)
        self.E_pot.append(E_p)


@dataclass
class History:
    """
    Represents data of interest which should be saved during the simulation
    """
    p_hist_list: List[ParticleHistory]
    E_kin: List[float]
    E_pot: List[float]

    def __init__(self, sys: System):
        self.p_hist_list = []
        for i, el in enumerate(sys.p_list):
            self.p_hist_list.append(ParticleHistory(i, sys))
        self.E_kin = []
        self.E_pot = []
        self.E_kin = [sum([p.E_kin[-1] for p in self.p_hist_list])]
        self.E_pot = [sum([p.E_pot[-1] for p in self.p_hist_list]) / 2]

    def record(self, sys: System):
        """
        Records current state of the system
        """
        for i, el in enumerate(sys.p_list):
            self.p_hist_list[i].record(i, sys)
        """
        kinetic energy of the system is the sum of kinetic energies of all
        particles in the system
        """
        self.E_kin.append(sum([p.E_kin[-1] for p in self.p_hist_list]))
        """
        potential energy of the system is the sum of interaction energies 
        related to each of the particles over 2 (to exclude double counting of 
        interactions)
        """
        self.E_pot.append(sum([p.E_pot[-1] for p in self.p_hist_list]) / 2)

"""
Classes dedicated to creating and saving graphs
"""
@dataclass(frozen=True)
class Plot:
    x: np.ndarray
    y: np.ndarray
    markers: str
    label: str


@dataclass(frozen=True)
class Figure:
    title: str
    xlabel: str
    ylabel: str
    xlim: List[float]
    ylim: List[float]
    legend_loc: str
    tex: bool
    plots: List[Plot]


def saveFigure(filename: str, *figures: Figure):
    """Save figures as pdf, pgf or png"""
    for j, figure in enumerate(figures):
        plt.rc("text", usetex=figure.tex)
        plt.figure(figsize=[8, 6])
        plt.title(figure.title)
        plt.xlabel(figure.xlabel)
        plt.ylabel(figure.ylabel)
        if(figure.xlim != [0, 0]):
            plt.xlim(figure.xlim)
        if(figure.ylim != [0, 0]):
            plt.ylim(figure.ylim)
        for plot in figure.plots:
            plt.plot(plot.x, plot.y, plot.markers, label=plot.label)
        plt.legend(loc=figure.legend_loc)
        plt.savefig(filename.split(".")[0] + "_" + str(j) + ".png", dpi=400, bbox_inches='tight')
        # plt.savefig(filename.split(".")[0] + "_" + str(j) + ".pgf") might require tex installation
        plt.close()


# Initial parameters of simulation
# use M☉, AU, yr for mass, distance, time units
dt = 1e-3
G = 4 * np.pi**2
M_sun = 2.0e30  # kg
e = 0.017
a = 1.00  # in AU
# perihelion position, speed
r_min = (1 - e) * a
v_r_min = np.sqrt(G / a * (1 + e) / (1 - e))
M_Earth = 6.0e24 / M_sun
# Initialize system and its history
particle_list = [Particle(1, np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])),
                 Particle(M_Earth, np.array([r_min, 0, 0]), np.array([0, v_r_min, 0]), np.array([0, 0, 0]))]
sys = System(dt, G, particle_list)
sys_history = History(sys)
t = 0
tmax = 2  # shift time axis so that tmin = 0
while t < tmax / dt:
    """
    Iterate system one time step forward until time tmax is reached
    """
    sys.update()
    sys_history.record(sys)
    t += 1

plotting_period = int(1 / dt / 100) # reduce memory used for 3d plot
r_mi = np.array([np.array([r_min, 0]) for i in range(t + 1)])[::plotting_period] # perihelion
r_ma = np.array([np.array([-r_min * (1 + e) / (1 - e), 0]) for i in range(t + 1)])[::plotting_period] # aphelion
t = np.linspace(0., t * dt, num=t + 1)
fig = plt.figure(figsize=(15, 15))
ax = plt.axes(projection='3d')
ax.set_xlabel('$t$ [yr]')
ax.set_ylabel('$x$ [AU]')
ax.set_zlabel('$y$ [AU]')
p0 = sys_history.p_hist_list[0]
p1 = sys_history.p_hist_list[1]
ax.scatter(t[::plotting_period], *zip(*[el[0:2] for el in p0.r][::plotting_period]), linewidth=0.1, label="Sun")
ax.scatter(t[::plotting_period], *zip(*[el[0:2] for el in p1.r][::plotting_period]), linewidth=0.1, label="Earth")
ax.scatter(t[::plotting_period], *zip(*r_mi), linewidth=0.1, label="Perihelion")
ax.scatter(t[::plotting_period], *zip(*r_ma), linewidth=0.1, label="Aphelion")
plt.title("Earth orbiting the Sun")
plt.legend(loc="upper right")
#plt.show() # show space-time graph of the system
saveFigure(
    "EnergyConservation.pdf", Figure(
        "Energy dependence on time", "$t$ [yr]", "$E(t)$ [Msol$\cdot$AU$^2$/yr$^2$]", [
            0, 0], [
                0, 0], "upper left", False, [
                    Plot(
                        t, sys_history.E_pot, "r:", "$E_p(t)$ calculated numerically"), Plot(
                            t, sys_history.E_kin, "g:", "$E_k(t)$ calculated numerically"), Plot(
                                t, np.array(
                                    sys_history.E_pot) + np.array(
                                        sys_history.E_kin), "b--", "$E(t)$ calculated numerically")]))
saveFigure("earthx.pdf",
           Figure("x-coordinate dependence on time", "$t$ [yr]", "$x(t)$ [AU]", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        [el[0] for el in sys_history.p_hist_list[1].r],
                        "r--",
                        "$x(t)$ calculated numerically")]),
           Figure("x-velocity dependence on time", "$t$ [yr]", "$v_x(t) [AU/yr]$", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        [el[0] for el in sys_history.p_hist_list[1].v],
                        "r--",
                        "$v_x(t)$ calculated numerically")]))
saveFigure("earthy.pdf",
           Figure("y-coordinate dependence on time", "$t$ [yr]", "$y(t) [AU]$", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        [el[1] for el in sys_history.p_hist_list[1].r],
                        "r--",
                        "$y(t)$ calculated numerically")]),
           Figure("y-velocity dependence on time", "$t$ [yr]", "$v_y(t)$ [AU/yr]", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        [el[1] for el in sys_history.p_hist_list[1].v],
                        "r--",
                        "$v_y(t)$ calculated numerically")]))
saveFigure("earth.pdf",
           Figure("distance to sun dependence on time", "$t$ [yr]", "$r(t) [AU]$", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        [np.sqrt(el[0]**2 + el[1]**2) for el in sys_history.p_hist_list[1].r],
                        "r--",
                        "$r(t)$ calculated numerically")]),
           Figure("velocity magnitude dependence on time", "$t$ [yr]", "$v(t)$ [AU/yr]", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        [np.sqrt(el[0]**2 + el[1]**2) for el in sys_history.p_hist_list[1].v],
                        "r--",
                        "$v(t)$ calculated numerically")]))
