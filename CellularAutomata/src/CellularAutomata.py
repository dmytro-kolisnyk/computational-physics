"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import List
from random import choice

# Create figure for plotting, set the fps and dpi of animation
fig, ax = plt.subplots()
fps = 10
dpi = 400  # 400


@dataclass
class System:
    """
    Cellular automaton system
    -------------------------
    cells : array of cells with boundary cells being barrier elements of array
    finite : allows to choose boundary conditions (True for bounded, False for periodic)
    trajectory : saves trajectory of the system
    """
    cells: np.ndarray[np.uint8]
    finite: bool
    trajectory: List

    def __init__(self, cells: np.ndarray[np.uint8], rule: np.uint8, finite: bool) -> None:
        """
        Initiates system given initial cells, system evolution rule and boundary conditions
        """
        self.cells = np.empty([el + 2 for el in cells.shape], np.uint8)
        for i, el in enumerate(cells):
            self.cells[i + 1] = el
        if(finite):
            self.cells[0] = np.uint8(0)
            self.cells[-1] = np.uint8(0)
        else:
            self.cells[0] = self.cells[-2]
            self.cells[-1] = self.cells[1]
        self.rule = rule
        self.finite = finite
        self.trajectory = [self.cells]

    def update(self):
        """
        Updates system according to the rule
        """
        cells = np.empty_like(self.cells)
        for i in range(1, len(self.cells) - 1):
            point_state = (self.cells[i - 1] << 2) + (self.cells[i] << 1) + self.cells[i + 1]
            cells[i] = (self.rule & (np.uint8(1) << point_state)) >> point_state
        cells[0] = np.uint8(0)
        cells[-1] = np.uint8(0)
        if(not self.finite):
            cells[0] = cells[-2]
            cells[-1] = cells[1]
        self.cells = cells
        self.trajectory.append(self.cells)


def init_animation():
    """Initiates figure"""
    ax.clear()
    ax.axis('off')
    return ax,


def animate(i, traj):
    i = i // fps
    """Grid vizualization"""
    ax.clear()
    arr = traj[i]
    arr = np.expand_dims(arr[1:-1], axis=0)  # vizuals w/o grid's barrier elements
    ax.imshow(arr, aspect=1)
    ax.axis('off')
    return ax,


def simulate(sys: System, iterations: int, fpsfactor: float, filename: str):
    """
    Simulates evolution of system for given amount of iterations
    """
    for i in range(iterations):
        sys.update()
    print("Rule",sys.rule,"finite bc:",sys.finite,"long run state: ",sys.cells[1:-1])
    # now system can be returned and its properties can be in principle further analysed in program
    # for current project, animation of evolution of system was recorded (see commented section below)
    """ani = FuncAnimation(fig, animate, frames=fps * (len(sys.trajectory) - 2), fargs=(
        sys.trajectory,), init_func=init_animation, repeat=False)
    writervideo = FFMpegWriter(fps=fpsfactor * fps)
    ani.save(filename, writer=writervideo, dpi=dpi)""" # Requires FFMpeg installation


# set number of iterations to record and the speedup factor of the resulting video
iterations_num = 200  # 200
fpsfactor = 2

# b.1
N = 100
cells = np.zeros([N], dtype=np.uint8)
cells[0] = np.uint8(1)
sys = System(cells, np.uint8(150), True)
simulate(sys, iterations_num, fpsfactor, "b_systemx150bounded.mp4")

# b.2
N = 100
cells = np.zeros([N], dtype=np.uint8)
cells[0] = np.uint8(1)
sys = System(cells, np.uint8(150), False)
simulate(sys, iterations_num, fpsfactor, "b_systemx150periodic.mp4")

# c.1
N = 100
cells = np.array([choice([np.uint8(0), np.uint8(1)]) for i in range(N)], dtype=np.uint8)
sys = System(cells, np.uint8(90), True)
simulate(sys, iterations_num, fpsfactor, "c_systemx90bounded.mp4")

# c.2
N = 100
cells = np.array([choice([np.uint8(0), np.uint8(1)]) for i in range(N)], dtype=np.uint8)
sys = System(cells, np.uint8(90), False)
simulate(sys, iterations_num, fpsfactor, "c_systemx90periodic.mp4")

# c.3
N = 100
cells = np.array([choice([np.uint8(0), np.uint8(1)]) for i in range(N)], dtype=np.uint8)
sys = System(cells, np.uint8(150), True)
simulate(sys, iterations_num, fpsfactor, "c_systemx150bounded.mp4")

# c.4
N = 100
cells = np.array([choice([np.uint8(0), np.uint8(1)]) for i in range(N)], dtype=np.uint8)
sys = System(cells, np.uint8(150), False)
simulate(sys, iterations_num, fpsfactor, "c_systemx150periodic.mp4")


# c.5
N = 100
cells = np.array([choice([np.uint8(0), np.uint8(1)]) for i in range(N)], dtype=np.uint8)
sys = System(cells, np.uint8(18), True)
simulate(sys, iterations_num, fpsfactor, "c_systemx18bounded.mp4")

# c.6
N = 100
cells = np.array([choice([np.uint8(0), np.uint8(1)]) for i in range(N)], dtype=np.uint8)
sys = System(cells, np.uint8(18), False)
simulate(sys, iterations_num, fpsfactor, "c_systemx18periodic.mp4")

# c.7
N = 100
cells = np.array([choice([np.uint8(0), np.uint8(1)]) for i in range(N)], dtype=np.uint8)
sys = System(cells, np.uint8(73), True)
simulate(sys, iterations_num, fpsfactor, "c_systemx73bounded.mp4")

# c.8
N = 100
cells = np.array([choice([np.uint8(0), np.uint8(1)]) for i in range(N)], dtype=np.uint8)
sys = System(cells, np.uint8(73), False)
simulate(sys, iterations_num, fpsfactor, "c_systemx73periodic.mp4")
# c.9
N = 100
cells = np.array([choice([np.uint8(0), np.uint8(1)]) for i in range(N)], dtype=np.uint8)
sys = System(cells, np.uint8(136), True)
simulate(sys, iterations_num, fpsfactor, "c_systemx136bounded.mp4")

# c.10
N = 100
cells = np.array([choice([np.uint8(0), np.uint8(1)]) for i in range(N)], dtype=np.uint8)
sys = System(cells, np.uint8(136), False)
simulate(sys, iterations_num, fpsfactor, "c_systemx136periodic.mp4")
# d.1
N = 100
cells = np.array([np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(0),
                  np.uint8(0), np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1),
                  np.uint8(1), np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(0),
                  np.uint8(0)] + [choice([np.uint8(0), np.uint8(1)]) for i in range(18, N)], dtype=np.uint8)
sys = System(cells, np.uint8(184), False)
simulate(sys, iterations_num, fpsfactor, "d_systemx184periodic1x3.mp4")

# d.2
N = 100
cells = np.array([np.uint8(0), np.uint8(0), np.uint8(0), np.uint8(1), np.uint8(1),
                  np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1),
                  np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(0), np.uint8(0),
                  np.uint8(0)] + [choice([np.uint8(0), np.uint8(1)]) for i in range(18, N)], dtype=np.uint8)
sys = System(cells, np.uint8(184), False)
simulate(sys, iterations_num, fpsfactor, "d_systemx184periodic2x3.mp4")

# d.3
N = 100
cells = np.array([np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1),
                  np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1),
                  np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1), np.uint8(1),
                  np.uint8(1)] + [choice([np.uint8(0), np.uint8(1)]) for i in range(18, N)], dtype=np.uint8)
sys = System(cells, np.uint8(184), False)
simulate(sys, iterations_num, fpsfactor, "d_systemx184periodic3x3.mp4")


# freestyle (e1)
N = 100
cells = np.zeros([N], dtype=np.uint8)
cells[1] = np.uint8(1)
cells[3] = np.uint8(1)
cells[-10] = np.uint8(1)
sys = System(cells, np.uint8(184), False)
simulate(sys, iterations_num, fpsfactor, "e_systemx184periodiclow.mp4")

# freestyle (e2)
N = 100
cells = np.zeros([N], dtype=np.uint8)
cells[-8] = np.uint8(1)
cells[-9] = np.uint8(1)
cells[-10] = np.uint8(1)
sys = System(cells, np.uint8(184), True)
simulate(sys, iterations_num, fpsfactor, "e_systemx184boundedcluster.mp4")
