"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7  

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import matplotlib.pyplot as plt
"""
For exporting graphs as pgf files

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
"""

@dataclass
class System:
    dt: float
    R: float
    G: float
    M: float
    k: float
    m: float
    r: np.ndarray
    v: np.ndarray
    a: np.ndarray

    def F_over_m(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.G * self.M / ((self.R + r[2])**2) * \
            np.array([0, 0, -1]) - self.k * np.linalg.norm(v) * v / self.m

    def update(self):
        """Euler scheme"""
        self.a = self.F_over_m(self.r, self.v)
        self.v = self.v + self.a * self.dt
        self.r = self.r + self.v * self.dt
    '''

    def update(self):
        """Runge-Kutta(4) scheme"""
        kv1 = self.F_over_m(self.r, self.v) * dt
        kx1 = self.v * dt
        kv2 = self.F_over_m(self.r + kx1 / 2, self.v + kv1 / 2) * dt
        kx2 = (self.v + kv1 / 2) * dt
        kv3 = self.F_over_m(self.r + kx2 / 2, self.v + kv2 / 2) * dt
        kx3 = (self.v + kv2 / 2) * dt
        kv4 = self.F_over_m(self.r + kx3 / 2, self.v + kv3) * dt
        kx4 = (self.v + kv3) * dt
        self.v = self.v + 1 / 6 * (kv1 + 2 * kv2 + 2 * kv3 + kv4)
        self.r = self.r + 1 / 6 * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
        self.a = self.F_over_m(self.r, self.v)
    '''

    def __init__(self, dt: float, R: float, G: float, M: float, k: float, m: float,
                 r: np.ndarray, v: np.ndarray):
        self.dt = dt
        self.G = G
        self.M = M
        self.R = R
        self.k = k
        self.m = m
        self.r = r
        self.v = v
        self.a = self.F_over_m(r,v)


@dataclass
class History:
    r: list
    v: list
    a: list

    def __init__(self, sys: System):
        self.r = [sys.r]
        self.v = [sys.v]
        self.a = [sys.a]


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
dt = 1e-5
G = 6.67e-11
R = 6370e3
M = 5.99e24
g = G * M / R**2
k = 1e-4
m = 50
r0 = np.array([0, 0, 5000])
v0 = np.array([0, 0, 0])

# Initialize system and its history
sys = System(dt, R, G, M, k, m, r0, v0)
sys_history = History(sys)
t = 0

while sys.r[2] > 0:
    sys.update()
    sys_history.r.append(sys.r)
    sys_history.v.append(sys.v)
    sys_history.a.append(sys.a)
    t += 1
t = np.linspace(0., t * dt, num=t + 1)

saveFigure("FallingBody.pdf",
           Figure("z-coordinate dependence on time", "$t$", "$z(t)$", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        [el[2] for el in sys_history.r],
                        "r--",
                        "$z(t)$ calculated numerically")]),
           Figure("z-velocity dependence on time", "$t$", "$v(t)$", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        [el[2] for el in sys_history.v],
                        "r--",
                        "$v_z(t)$ calculated numerically")]),
           Figure("z-acceleration dependence on time", "$t$", "$a(t)$", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        [el[2] for el in sys_history.a],
                        "r--",
                        "$a_z(t)$ calculated numerically")]))
