"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pylab as plt
import itertools


@dataclass
class Lattice:
    """Simulated lattice, on which potential distribution is calculated"""
    V: np.ndarray
    rho: np.ndarray
    h: float

    def calc_error(self, old: np.ndarray, new: np.ndarray):
        """Error of convergence - sum over all lattice points of |V_new[i][j]-V_old[i][j]|"""
        rows, cols = old.shape
        return sum([np.abs(new[i, j] - old[i, j])
                    for (i, j) in itertools.product(range(1, rows - 1), range(1, cols - 1))])

    def upd_Jacobi(self):
        """Jacobi method"""
        V_new = np.empty_like(self.V)
        rows, cols = self.V.shape
        for (i, j) in itertools.product(range(0, rows), range(0, cols)):
            if ((i == 0 or i == self.V.shape[0] - 1) or (j == 0 or j == self.V.shape[1] - 1)):
                V_new[i, j] = self.V[i, j]
            else:
                V_new[i, j] = 1 / 4 * (self.V[i + 1, j] + self.V[i - 1, j] + self.V[i, j + 1] +
                                       self.V[i, j - 1] + self.h**2 * self.rho[i, j])
        error = self.calc_error(self.V, V_new)
        self.V = V_new
        return error

    def upd_GaussSeidel(self):
        """Gauss-Seidel method"""
        V_new = np.empty_like(self.V)
        rows, cols = self.V.shape
        for (i, j) in itertools.product(range(0, rows), range(0, cols)):
            if ((i == 0 or i == self.V.shape[0] - 1) or (j == 0 or j == self.V.shape[1] - 1)):
                V_new[i, j] = self.V[i, j]
            else:
                V_new[i, j] = 1 / 4 * (self.V[i + 1, j] + self.V[i, j + 1] + V_new[i - 1, j] +
                                       V_new[i, j - 1] + self.h**2 * self.rho[i, j])
        error = self.calc_error(self.V, V_new)
        self.V = V_new
        return error

    def upd_SOR(self, w: float):
        """Successive over relaxation method"""
        V_new = np.empty_like(self.V)
        rows, cols = self.V.shape
        for (i, j) in itertools.product(range(0, rows), range(0, cols)):
            if ((i == 0 or i == self.V.shape[0] - 1) or (j == 0 or j == self.V.shape[1] - 1)):
                V_new[i, j] = self.V[i, j]
            else:
                V_new[i, j] = (1 - w) * self.V[i, j] + w * 1 / 4 * (self.V[i + 1, j] + self.V[i, j + 1] + \
                               V_new[i - 1, j] + V_new[i, j - 1] + self.h**2 * self.rho[i, j])
        error = self.calc_error(self.V, V_new)
        self.V = V_new
        return error


def heatmap2d(arr: np.ndarray):
    """Lattice vizualization"""
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()


# a.1
# set initial state of the system
Lx = 11
Ly = 21
gridsize = 1
V = np.ndarray((Lx, Ly))
for (i, j) in itertools.product(range(0, Lx), range(0, Ly)):
    if ((i == 0 or i == V.shape[0] - 1) or (j == 0 or j == V.shape[1] - 1)):
        V[i, j] = 2
    else:
        V[i, j] = 1  # starting guess (!=2) for initial value of potential
rho = np.full((Lx, Ly), 0)
a_system = Lattice(V, rho, gridsize)
i = 0
while a_system.upd_Jacobi() > 1e-4:
    i += 1
print("# of Jacobi iterations for 1e-4 precision (h=1):", i)

# a.2
# set initial state of the system
Lx = 11
Ly = 21
gridsize = 0.5  # grid size reduced by a factor of 2
V = np.ndarray((Lx, Ly))
for (i, j) in itertools.product(range(0, Lx), range(0, Ly)):
    if ((i == 0 or i == V.shape[0] - 1) or (j == 0 or j == V.shape[1] - 1)):
        V[i, j] = 2
    else:
        V[i, j] = 1
rho = np.full((Lx, Ly), 0)
a_system = Lattice(V, rho, gridsize)
i = 0
while a_system.upd_Jacobi() > 1e-4:
    i += 1
print("# of Jacobi iterations for 1e-4 precision (h=0.5):", i)


# b
# set initial state of the system
Lx = 11
Ly = 21
gridsize = 1
V = np.ndarray((Lx, Ly))
for (i, j) in itertools.product(range(0, Lx), range(0, Ly)):
    if ((i == 0 or i == V.shape[0] - 1) or (j == 0 or j == V.shape[1] - 1)):
        V[i, j] = 2
    else:
        V[i, j] = 1  # starting guess (!=2) for initial value of potential
rho = np.full((Lx, Ly), 0)
rho[Lx // 2, Ly // 2] = 5
a_system = Lattice(V, rho, gridsize)
i = 0
while a_system.upd_Jacobi() > 1e-4:
    i += 1
print("# of Jacobi iterations for 1e-4 precision (rho==5delta):", i)
heatmap2d(a_system.V)
# c.1
# set initial state of the system
Lx = 11
Ly = 21
gridsize = 1
V = np.ndarray((Lx, Ly))
for (i, j) in itertools.product(range(0, Lx), range(0, Ly)):
    if ((i == 0 or i == V.shape[0] - 1) or (j == 0 or j == V.shape[1] - 1)):
        V[i, j] = 2
    else:
        V[i, j] = 1  # starting guess (!=2) for initial value of potential
rho = np.full((Lx, Ly), 0)
a_system = Lattice(V, rho, gridsize)
i = 0
while a_system.upd_GaussSeidel() > 1e-4:
    i += 1
print("# of GaussSeidel iterations for 1e-4 precision (h=1):", i)

# c.2
# set initial state of the system
Lx = 11
Ly = 21
gridsize = 0.5  # grid size reduced by a factor of 2
V = np.ndarray((Lx, Ly))
for (i, j) in itertools.product(range(0, Lx), range(0, Ly)):
    if ((i == 0 or i == V.shape[0] - 1) or (j == 0 or j == V.shape[1] - 1)):
        V[i, j] = 2
    else:
        V[i, j] = 1
rho = np.full((Lx, Ly), 0)
a_system = Lattice(V, rho, gridsize)
i = 0
while a_system.upd_GaussSeidel() > 1e-4:
    i += 1
print("# of GaussSeidel iterations for 1e-4 precision (h=0.5):", i)

# d-a.1
# set initial state of the system
Lx = 11
Ly = 21
gridsize = 1
V = np.ndarray((Lx, Ly))
for (i, j) in itertools.product(range(0, Lx), range(0, Ly)):
    if ((i == 0 or i == V.shape[0] - 1) or (j == 0 or j == V.shape[1] - 1)):
        V[i, j] = 2
    else:
        V[i, j] = 1  # starting guess (!=2) for initial value of potential
rho = np.full((Lx, Ly), 0)
a_system = Lattice(V, rho, gridsize)
i = 0
while a_system.upd_SOR(2 / (1 + 2 * np.pi / (Lx + Ly))) > 1e-4:
    i += 1
print("# of SOR iterations for 1e-4 precision (h=1):", i)

# d-a.2
# set initial state of the system
Lx = 11
Ly = 21
gridsize = 0.5  # grid size reduced by a factor of 2
V = np.ndarray((Lx, Ly))
for (i, j) in itertools.product(range(0, Lx), range(0, Ly)):
    if ((i == 0 or i == V.shape[0] - 1) or (j == 0 or j == V.shape[1] - 1)):
        V[i, j] = 2
    else:
        V[i, j] = 1
rho = np.full((Lx, Ly), 0)
a_system = Lattice(V, rho, gridsize)
i = 0
while a_system.upd_SOR(2 / (1 + 2 * np.pi / (Lx + Ly))) > 1e-4:
    i += 1
print("# of SOR iterations for 1e-4 precision (h=0.5):", i)


# d-b
# set initial state of the system
Lx = 11
Ly = 21
gridsize = 1
V = np.ndarray((Lx, Ly))
for (i, j) in itertools.product(range(0, Lx), range(0, Ly)):
    if ((i == 0 or i == V.shape[0] - 1) or (j == 0 or j == V.shape[1] - 1)):
        V[i, j] = 2
    else:
        V[i, j] = 1  # starting guess (!=2) for initial value of potential
rho = np.full((Lx, Ly), 0)
rho[Lx // 2, Ly // 2] = 5
a_system = Lattice(V, rho, gridsize)
i = 0
while a_system.upd_SOR(2 / (1 + 2 * np.pi / (Lx + Ly))) > 1e-4:
    i += 1
print("# of SOR iterations for 1e-4 precision (rho=5delta):", i)

# d-free_init_conditions
# set initial state of the system
Lx = 201
Ly = 201
gridsize = 1
V = np.ndarray((Lx, Ly))
for (i, j) in itertools.product(range(0, Lx), range(0, Ly)):
    if ((i == 0 or i == V.shape[0] - 1) or (j == 0 or j == V.shape[1] - 1)):
        V[i, j] = 0
    else:
        V[i, j] = 0  # starting guess (!=2) for initial value of potential
rho = np.full((Lx, Ly), 0)
rho[Lx // 4, Ly // 2] = 1
rho[3*(Lx // 4), Ly // 2] = -1
a_system = Lattice(V, rho, gridsize)
i = 0
while a_system.upd_SOR(2 / (1 + 2 * np.pi / (Lx + Ly))) > 1e-4:
    i += 1
print("# of SOR iterations for 1e-4 precision (rho=1delta, Lx=Ly=201):", i)
heatmap2d(a_system.V)
