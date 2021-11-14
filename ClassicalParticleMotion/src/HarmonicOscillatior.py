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
    k: float
    m: float
    r: np.ndarray
    v: np.ndarray
    a: np.ndarray

    def F_over_m(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        return - self.k * r / self.m

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

    def __init__(self, dt: float, k: float, m: float,
                 r: np.ndarray, v: np.ndarray):
        self.dt = dt
        self.k = k
        self.m = m
        self.r = r
        self.v = v
        self.a = self.F_over_m(self.r, self.v)


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


def xTheor(k: float, m: float, t: np.ndarray, r0: np.ndarray):
    return r0[0] * np.cos(t * np.sqrt(k / m))


def vTheor(k: float, m: float, t: np.ndarray, r0: np.ndarray):
    return -r0[0] * np.sqrt(k / m) * np.sin(t * np.sqrt(k / m))


def aTheor(k: float, m: float, t: np.ndarray, r0: np.ndarray):
    return -r0[0] * k / m * np.cos(t * np.sqrt(k / m))

# Initial parameters of simulation
dt = 1e-4
k = 1
m = 1
r0 = np.array([0.5, 0, 0])
v0 = np.array([0, 0, 0])

# Initialize system and its history
sys = System(dt, k, m, r0, v0)
sys_history = History(sys)
t = 0

# 495 oscillations
while t < 3300 / dt:
    sys.update()
    sys_history.r.append(sys.r)
    sys_history.v.append(sys.v)
    sys_history.a.append(sys.a)
    t += 1
t = np.linspace(0., t * dt, num=t + 1)

# print numerical errors
print(sys_history.r[-1]," Del x =",sys_history.r[-1][0]-xTheor(k, m, t, r0)[-1]," varepsilon x =",(sys_history.r[-1][0]-xTheor(k, m, t, r0)[-1])/sys_history.r[-1][0])
print(sys_history.v[-1]," Del v =",sys_history.v[-1][0]-vTheor(k, m, t, r0)[-1]," varepsilon v =",(sys_history.v[-1][0]-vTheor(k, m, t, r0)[-1])/sys_history.v[-1][0])
print(sys_history.a[-1]," Del a =",sys_history.a[-1][0]-aTheor(k, m, t, r0)[-1]," varepsilon a =",(sys_history.a[-1][0]-aTheor(k, m, t, r0)[-1])/sys_history.a[-1][0])

saveFigure("HarmonicOscillator.pdf",
           Figure("x-coordinate dependence on time", "$t$", "$x(t)$", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        xTheor(k, m, t, r0),
                        "b",
                        "$x(t)$ calculated theoretically"),
                   Plot(t,
                        [el[0] for el in sys_history.r],
                        "r--",
                        "$x(t)$ calculated numerically")]),
           Figure("x-velocity dependence on time", "$t$", "$v_x(t)$", [0, 0], [0, 0],
                  "upper right", False,
                  [Plot(t,
                        vTheor(k, m, t, r0),
                        "b",
                        "$v_x(t)$ calculated theoretically"),
                   Plot(t,
                        [el[0] for el in sys_history.v],
                        "r--",
                        "$v_x(t)$ calculated numerically")]),
           Figure("x-acceleration dependence on time", "$t$", "$a_x(t)$", [0, 0], [0, 0],
                  "upper right", False,
                  [
               Plot(t,
                    aTheor(k, m, t, r0),
                    "b",
                    "$a_x(t)$ calculated theoretically"),
               Plot(t,
                    [el[0] for el in sys_history.a],
                    "r--",
                    "$a_x(t)$ calculated numerically")]))
