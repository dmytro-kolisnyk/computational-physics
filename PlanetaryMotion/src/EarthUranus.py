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


def runSim(dt):
    # Initial parameters of simulation
    # use M☉, AU, yr for mass, distance, time units
    G = 4 * np.pi**2
    M_sun = 2.0e30  # kg
    e_Earth = 0.017
    a_Earth = 1.00  # in AU
    r_min_Earth = (1 - e_Earth) * a_Earth
    v_r_min_Earth = np.sqrt(G / a_Earth * (1 + e_Earth) / (1 - e_Earth))
    e_Uranus = 0.046
    a_Uranus = 19.19  # in AU
    r_max_Uranus = (1 + e_Uranus) * a_Uranus
    v_r_max_Uranus = np.sqrt(G / a_Uranus * (1 - e_Uranus) / (1 + e_Uranus))
    # Initialize system and its history
    particle_list = [Particle(1, np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])),
                     Particle(6.0e24 / M_sun, np.array([r_min_Earth, 0, 0]), np.array([0, v_r_min_Earth, 0]), np.array([0, 0, 0])),
                     Particle(8.8e25 / M_sun, np.array([-r_max_Uranus, 0, 0]), np.array([0, -v_r_max_Uranus, 0]), np.array([0, 0, 0]))]
    sys = System(dt, G, particle_list)
    sys_history = History(sys)
    t = 0

    while t < 84 / dt:  # one uranus orbital period
        sys.update()
        sys_history.record(sys)
        t += 1
    E = np.array(sys_history.E_pot) + np.array(sys_history.E_kin)
    return np.abs(E[-1] - E[0])


y = []
x = np.linspace(1e-4, 1, 20)
for dt in x:
    y.append(runSim(dt))
plt.plot(x, y, 'bo')
plt.xlabel("$dt$ [yr]", fontsize=18)
plt.ylabel(
    r"$|E_1-E_0|$ [M☉$\cdot$AU$^2$/yr$^2$] where" +
    "\n $E_1$ is total energy of the system after 1 Uranus orbit\n$E_0$ is starting total energy of the system",
    fontsize=18)
plt.show()
