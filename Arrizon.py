import numpy as np
# import cupy as cp
import scipy
import matplotlib.pyplot as plt
import matplotlib

import WGS
import slm
import profile

J0 = np.flip(np.load('J0.npy'), axis=1)


def bisection(array, value):
    """Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively."""
    n = len(array)

    if value < array[0]:
        return -1
    elif value > array[n - 1]:
        return n
    jl = 0  # Initialize lower
    ju = n - 1  # and upper limits.
    while ju - jl > 1:  # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if value >= array[jm]:
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    elif value == array[n - 1]:  # and top
        return n - 1
    else:
        return jl


def invJ0(a):
    j = bisection(J0[1], a)
    if j < len(J0[1]) - 1 and np.abs(J0[1, j + 1] - a) < np.abs(J0[1, j] - a):
        j += 1
    return J0[0, j]


def write_J0(interval, num):
    x = np.linspace(interval[0], interval[1], num)
    J0 = scipy.special.j0(x)
    table = np.array([x, J0])

    np.save('J0.npy', table)


def load_J0():
    return np.flip(np.load('J0.npy'), axis=1)


def type2_psi_nm(a, phi):
    return phi + np.vectorize(pyfunc=invJ0, otypes=[np.float64])(a) * np.sin(phi)


if __name__ == '__main__':
    # write_J0([0, scipy.special.jn_zeros(0, 1)[0]], int(1e6))
    # print('done writing')

    slm = slm.SLM()

    # J0 = load_J0()
    # print(J0)

    plt.plot(J0[1], J0[0])
    plt.title('Inverse zero-order bessel function')
    plt.show()

    target, spots = profile.Profile.gaussian_array(1, 5)

    target = np.array(target, dtype=np.complex128)
    # target *= np.exp(1j * slm.half())

    slm.ampToBMP(np.abs(target), name='1x5_target_amp', color=True)
    slm.phaseToBMP(np.angle(target), name='1x5_target_phase', color=True)

    a = np.abs(target)
    phi = np.angle(target)

    psi = type2_psi_nm(a, phi)

    # slm.phaseToBMP(psi, name='1x5_input_phase', color=True)

    image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * psi)), norm="ortho"))

    input_amp = profile.Profile.input_gaussian(beam_size=(0.05, 0.05))

    # slm.ampToBMP(np.abs(input_amp), name='1x5_input_amp', color=True)
    slm.phaseToBMP(psi, name='1x5_input_phase', color=True)

    # input_prof = np.abs(input_amp) * np.exp(1j * (np.angle(image)))

    # image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image), norm="ortho"))

    slm.ampToBMP(np.abs(image), name='1x5_output_amp', color=True)
    slm.phaseToBMP(np.angle(image), name='1x5_output_phase', color=True)
