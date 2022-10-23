"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
import numpy as np
from scipy import stats
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


def linear_congruential_rng(x, a, b, c):
    """Generate next element of the linear congruential rng sequence"""
    return (x * a + b) % c


def generate_sequence(x0, n, step_function):
    """Generate random sequence using linear congruential rng parameters"""
    random_sequence = [x0]
    for i in range(n):
        random_sequence.append(step_function(random_sequence[-1]))
    return random_sequence


def find_period(sequence):
    """Return length of first gap in a sequence (period)"""
    table = {}
    for i, el in enumerate(sequence):
        if el in table:
            return i - table[el]
        else:
            table[el] = i
    return -1


def rho(seq, i, j):
    """Return autocorrelation function"""
    N = len(seq)
    mean = np.mean(seq)
    seq = [el - mean for el in seq]
    M = int(np.floor((N - 1 - i) / j) - 1)
    if(M < 1):
        print("Error: Larger sample space/smaller lag needed")
        exit(1)
    autocor = sum([seq[i + k * j] * seq[i + (k + 1) * j] for k in range(M + 1)]) / np.sqrt(sum([seq[i + k * j]
                                                                                                ** 2 for k in range(M + 1)])) / np.sqrt(sum([seq[i + (k + 1) * j]**2 for k in range(M + 1)]))
    return autocor


def gaps_num(seq, interval):
    """Find gap lengths starting and ending in a given interval"""
    i = 0
    gaps = []
    while i < len(seq):
        while (i < len(seq)) and (not(interval[0] <= seq[i] <= interval[1])):
            i += 1
        i += 1
        counter = 0
        while (i < len(seq)) and (not(interval[0] <= seq[i] <= interval[1])):
            i += 1
            counter += 1
        if(i < len(seq) - 1):
            gaps.append(counter)
    return gaps


def one_time_pad(msg, rng_seq):
    """Encode string into another string using random sequence"""
    msg = [msg[i:i + 8] for i in range(0, len(msg), 8)]
    res = ""
    for i, s in enumerate(msg):
        res += chr(int(s,2) ^ (rng_seq[i] % 256))
    return res


def text_to_bin(txt):
    """Convert string to binary"""
    return "".join(f"{ord(x):08b}" for x in txt)


def bin_str_to_ascii_text(b_str):
    """Convert binary to text if decodable"""
    b_str = int(b_str, 2)
    binary_array = b_str.to_bytes(b_str.bit_length(), "big")
    return binary_array.decode()

# first lcrng
a, b, c, x0 = 57, 1, 256, 10
sequence_index = c
sequence = generate_sequence(x0, sequence_index, lambda z: linear_congruential_rng(z, a, b, c))
print(sequence[:11], "...", sequence[-5:])
print("Period:", find_period(sequence))
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Distribution of neighbouring pairs of numbers in the random" +
              "\n" + "sequence, generated using $a, b, c, x_0 = 57, 1, 256, 10$")
axs.set_xlabel("$x_i$")
axs.set_ylabel("$x_{i+1}$")
axs.scatter(sequence[:-1], sequence[1:], c=np.arange(len(sequence) - 1), cmap='winter')
if(not DEBUG):
    plt.savefig("../res/ScatterPlot.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/ScatterPlot.png", bbox_inches="tight", format="png")
# custom lcrng
key = 10
a, b, c, x0 = 15005, 8371, 19993, key
sequence_index = c
sequence = generate_sequence(x0, sequence_index, lambda z: linear_congruential_rng(z, a, b, c))
uniform_seq = [el / (c - 1) for el in sequence]
# perform K-S test
print(stats.kstest(uniform_seq, 'uniform'))
# perform autocorrelation test
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
l = 1000
axs.set_title("Distribution of autocorrelations " + r"$\rho_{il}$" + f" with $l={l}$")
autocorrelations = []
for i in range(l):
    autocorrelations.append(rho(uniform_seq, i, l))
axs.set_xlabel("Correlation value")
axs.set_ylabel("PDF")
_, bins, _ = axs.hist(autocorrelations, bins=20, density=True, label="Empirical data")
mu, sigma = stats.norm.fit(autocorrelations)
best_fit_line = stats.norm.pdf(bins, mu, sigma)
axs.plot(bins, best_fit_line, label="Gaussian fit")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/AutoCorrelations.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/AutoCorrelations.png", bbox_inches="tight", format="png")
# perform gap test
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Distribution of gap sizes")
axs.set_xlabel("Gap size value")
axs.set_ylabel("PDF")
gap_sizes = []
interval_step = 0.01
for i in np.arange(interval_step, 1 + interval_step, interval_step):
    gap_sizes = [*gap_sizes, *gaps_num(uniform_seq, [i - interval_step, i])]
axs.hist(gap_sizes, bins=100, density=True, label="Empirical data")
x_range = np.arange(0, 1000, 10)
q = 1 - interval_step
axs.plot(x_range, [-np.log(q) * np.power(q, x) for x in x_range], label=r"PDF$(x)= -\ln{(1-p)}\cdot (1-p)^x$")
plt.legend()
if(not DEBUG):
    plt.savefig("../res/GapTest.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/GapTest.png", bbox_inches="tight", format="png")
# encode secret message
secret_msg = "Ckccubekc xdska haxbyb vlfkdvgv ff dhb oebcpf."
encr = one_time_pad(text_to_bin(secret_msg), sequence)
print("Encrytped binary string",text_to_bin(encr))
# encr = input("Enter a binary string to decrypt:")
encr = text_to_bin(encr) # input encrypted binary string back to decode
# decrypt encoded bitstring
# print(one_time_pad(encr, sequence))
if(DEBUG):
    plt.show()
