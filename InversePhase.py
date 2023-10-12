import numpy as np
# import cupy as cp
import scipy
import matplotlib.pyplot as plt
import matplotlib

# import WGS
import slm
import profile


def inverse_phase(slm, color=True, target=None, spots=None, input_size=(0.05, 0.05)):
    if target is None or spots is None:
        target, spots = profile.Profile.gaussian_array(1, 5)

    n = len(spots)

    # target = np.array(np.abs(target), dtype=np.complex128)

    # target = profile.Profile.input_gaussian(beam_size=(0.05, 0.05))
    # target *= np.exp(1j * slm.half())

    slm.ampToBMP(np.abs(target), name='1x%d_target_amp' % n, color=color)
    slm.phaseToBMP(np.angle(target), name='1x%d_target_phase' % n, color=color)

    # slm.phaseToBMP(psi, name='1x5_input_phase', color=True)

    # image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.exp(1j * psi)), norm="ortho"))
    ideal = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(target), norm="ortho"))
    slm.ampToBMP(np.abs(ideal), name='1x%d_ideal_amp' % n, color=color)
    slm.phaseToBMP(np.angle(ideal), name='1x%d_ideal_phase' % n, color=color, correction=False)

    input_amp = profile.Profile.input_gaussian(beam_size=input_size, size=slm.size)
    slm.ampToBMP(np.abs(input_amp), name='1x%d_input_amp' % n, color=color)

    actual = np.abs(input_amp) * np.exp(1j * np.angle(ideal))

    image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(actual), norm="ortho"))
    slm.ampToBMP(np.abs(image), name='1x%d_ideal_output_amp' % n, color=color)
    slm.phaseToBMP(np.angle(image), name='1x%d_ideal_output_phase' % n, color=color)

    return np.angle(ideal)


if __name__ == '__main__':
    size = np.array((1024, 1272)) * 1

    slm = slm.SLM(size=size)

    target, spots = profile.Profile.gaussian_array(1, 2, amps=np.exp(np.array([0, 0, 0, 0, 0]) * np.pi * 1j), y_pitch=0.004, waist=(0.002, 0.002), size=size)

    inverse_phase(slm, target=target, spots=spots, input_size=(0.3, 0.3))
