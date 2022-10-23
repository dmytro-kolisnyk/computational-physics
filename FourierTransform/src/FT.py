"""
Dmytro Kolisnyk d.kolisnyk@jacobs-university.de

Python 3.7.7
matplotlib-3.4.3

"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import wavfile
from scipy.fft import fft, ifft
from time import perf_counter
# setup plotting

plt.style.use('seaborn-pastel')
dpi = 400
# enables debug mode of the program (disconnects from additional external dependencies, e.g LaTeX, FFMpeg, ...)
DEBUG = True
if(not DEBUG):
    matplotlib.use('Agg')
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


def dft(f):
    """Manual DFT implementation"""
    N = f.shape[0]
    W = [np.exp(-2 * np.pi * 1j * n / N) for n in range(N)]
    res = []
    for k in range(N):
        res.append(np.dot(np.power(W, k), f))
    return np.array(res)


# read the audiofile
samplerate, dataA = wavfile.read('Voice.wav')
print("Sample rate:", samplerate)
# plot the sound signal
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Sound signal")
axs.set_xlabel(r"Time $t$ [s]")
axs.set_ylabel(r"Amplitude $f(t)$")
time = np.linspace(0., dataA.shape[0] / samplerate, dataA.shape[0])
axs.plot(time, dataA)
if(not DEBUG):
    plt.savefig("../res/datalong.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/datalong.png", bbox_inches="tight", format="png")
if(DEBUG):
    plt.show()
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
axs.set_title("Sound signal")
axs.set_xlabel(r"Time $t$ [s]")
axs.set_ylabel(r"Amplitude $f(t)$")
idmin = 48500
idmax = 53000
axs.plot(time[idmin:idmax], dataA[idmin:idmax])
if(not DEBUG):
    plt.savefig("../res/datashort.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/datashort.png", bbox_inches="tight", format="png")
if(DEBUG):
    plt.show()
N = idmax - idmin
T = N / samplerate
g = fft(dataA[idmin:idmax])
fig, axs = plt.subplots(1, 2, figsize=set_size(400, subplots=(1, 1)))
fig.suptitle(r"Power spectrum of the sound signal ($\omega_\mathrm{max}=4248$ [rad/s])")
axs[0].set_xlabel(r"Cyclic frequency $\omega$ [rad/s]")
axs[0].set_ylabel(r"Power $S(\omega)=\tilde{f}^*(\omega)\tilde{f}(\omega)$")
time = np.linspace(0., dataA.shape[0] / samplerate, dataA.shape[0])
omega = [2 * np.pi / T * k for k in range(idmax - idmin)]
axs[0].plot(omega, np.real(g.conjugate() * g), label="Scipy FFT")
axs[0].legend()
# find power spectrum
g = dft(dataA[idmin:idmax])
axs[1].set_xlabel(r"Cyclic frequency $\omega$ [rad/s]")
axs[1].set_ylabel(r"Power $S(\omega)=\tilde{f}^*(\omega)\tilde{f}(\omega)$")
axs[1].plot(omega, np.real(g.conjugate() * g), label="Manual DFT")
axs[1].legend()
if(not DEBUG):
    plt.savefig("../res/ftshort.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/ftshort.png", bbox_inches="tight", format="png")
if(DEBUG):
    plt.show()
# test the performance of algorithms (SciPy FFT vs simple DFT)
amax = 4
N = np.power(10, amax)
t_fft = []
t_dft = []
for a in range(1, amax + 1):
    idmax = idmin + np.power(10, a)
    t1_start = perf_counter()
    g = fft(dataA[idmin:idmax])
    t1_stop = perf_counter()
    t_fft.append(t1_stop - t1_start)
    t2_start = perf_counter()
    g = dft(dataA[idmin:idmax])
    t2_stop = perf_counter()
    t_dft.append(t2_stop - t2_start)
extra_fft = 3 # 1,...
for a in range(amax + 1, amax + extra_fft):
    idmax = idmin + np.power(10, a)
    t1_start = perf_counter()
    g = fft(dataA[idmin:idmax])
    t1_stop = perf_counter()
    t_fft.append(t1_stop - t1_start)
colors = ['dodgerblue', 'crimson']
fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
N_list = [np.power(10, a) for a in range(1, amax + 1)]
N_list_fft = [np.power(10, a) for a in range(1, amax + extra_fft)]
print("N_dft:", N_list)
print("N_fft:", N_list)
print("FFT time:", t_fft)
print("DFT time:", t_dft)
axs.loglog(N_list_fft, t_fft, label="Elapsed time for Scipy FFT", color=colors[0], alpha=0.5)
axs.loglog(N_list, t_dft, label="Elapsed time for Manual DFT", color=colors[1], alpha=0.5)
N_log = np.log10(np.array(N_list))
N_log_fft = np.log10(np.array(N_list_fft))
dft_log = np.log10(np.array(t_dft))
fft_log = np.log10(np.array(t_fft))
fft_id_min = 2
coefs_fft = np.polyfit(N_log_fft[fft_id_min:], fft_log[fft_id_min:], 1)
coefs_dft = np.polyfit(N_log, dft_log, 1)
axs.set_xlabel("Number of samples")
axs.set_ylabel("Execution time")
axs.set_title("Performance of different implementations of the Fourier transforms")
axs.loglog(N_list_fft[fft_id_min:], np.power(10, np.poly1d(coefs_fft)(N_log_fft[fft_id_min:])), "--", color=colors[0])
axs.loglog(N_list, np.power(10, np.poly1d(coefs_dft)(N_log)), "--", color=colors[1])
axs.legend()
print("FFT speed fit", coefs_fft)
print("DFT speed fit", coefs_dft)
if(not DEBUG):
    plt.savefig("../res/perf.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/perf.png", bbox_inches="tight", format="png")
if(DEBUG):
    plt.show()
# check that IFFT(FFT)=Id
g = fft(dataA)
fig, axs = plt.subplots(1, 2, figsize=set_size(400, subplots=(1, 1)))
fig.suptitle("FFT and IFFT(FFT) of the sound signal (real parts)")
axs[0].set_xlabel(r"Cyclic frequency $\omega$ [rad/s]")
axs[0].set_ylabel(r"$\mathcal{F}[f](\omega)$")
time = np.linspace(0., dataA.shape[0] / samplerate, dataA.shape[0])
omega = [2 * np.pi / T * k for k in range(dataA.shape[0])]
axs[0].plot(omega, np.real(g))
h = ifft(g)
axs[1].set_xlabel(r"Time $t$ [s]")
axs[1].set_ylabel(r"$\mathcal{F}^{-1}[\mathcal{F}[f]](t)$")
axs[1].plot(time, np.real(h))
if(not DEBUG):
    plt.savefig("../res/ftlong.pgf", bbox_inches="tight", format="pgf")
    plt.savefig("../res/ftlong.png", bbox_inches="tight", format="png")
if(DEBUG):
    plt.show()
# Compress audiofile
idmin = 48500
idmax = 53000
for a in range(1, 11):
    compress_num = int(np.ceil(dataA.shape[0] * (1 - np.power(0.5, a))))
    g_compressed = copy.deepcopy(g)
    for i in np.argpartition(np.real(g.conjugate() * g), compress_num)[:compress_num].tolist():
        g_compressed[i] = 0
    h_compressed = ifft(g_compressed)
    fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
    axs.set_title("Compressed sound signal ($f_a=1-0.5^{" + str(a) + "}$)")
    axs.set_xlabel(r"Time $t$ [s]")
    axs.set_ylabel(r"Amplitude $f(t)$")
    axs.plot(time[idmin:idmax], np.real(h_compressed[idmin:idmax]))
    if(not DEBUG):
        plt.savefig(f"../res/datashort_{a}.pgf", bbox_inches="tight", format="pgf")
        plt.savefig(f"../res/datashort_{a}.png", bbox_inches="tight", format="png")
    if(DEBUG):
        plt.show()
    wavfile.write(f"../sounds/compressed_{a}.wav", samplerate, np.real(h_compressed).astype(np.int16))
# Test lower sampling rate
idmin = 48500
idmax = 53000
for div_sample_num in [10, 100]:
    h = fft(dataA[idmin:idmax:div_sample_num])
    h = np.concatenate([h[:len(h) // 2],  np.zeros((div_sample_num - 1) *len(h)), h[len(h) // 2:]])
    q = ifft(h)
    time = np.linspace(0., dataA.shape[0] / samplerate, dataA.shape[0])
    fig, axs = plt.subplots(1, 1, figsize=set_size(400, subplots=(1, 1)))
    axs.set_title(f"Compressed sound signal (Sampling number reduced by a factor of {div_sample_num})")
    axs.set_xlabel(r"Time $t$ [s]")
    axs.set_ylabel(r"Amplitude $f(t)$")
    axs.plot(time[idmin:idmax],
             np.real(q))
    if(not DEBUG):
        plt.savefig(f"../res/datashort_samples{div_sample_num}.pgf", bbox_inches="tight", format="pgf")
        plt.savefig(f"../res/datashort_samples{div_sample_num}.png", bbox_inches="tight", format="png")
    if(DEBUG):
        plt.show()
