from PIL import Image as im
import numpy as np
from slm import SLM

pi = np.pi
size = np.array((1024, 1272))
lut = {413: 103}
correction_image = im.open("images/413corrwithLUT.bmp")
correction = np.array(correction_image) / lut[413] * 2 * pi


def half(center=0):
    center = np.array(size * (center / 2 + 0.5), np.uint)
    ar = np.zeros(size)
    ar[:, :center[1]] = pi
    return ar


def quad(center=np.array((0, 0))):
    center = np.array(size * (center / 2 + 0.5), np.uint)
    ar = np.zeros(size)
    ar[:center[0], :center[1]] = pi
    ar[center[0]:, center[1]:] = pi
    return ar


def add(a, b):
    out = (a + b) % (2 * pi)
    return out


def toBMP(phase_map, name='output', wavelength=413):
    bmp_array = np.array(phase_map / (2 * pi) * lut[wavelength], dtype=np.uint8)
    im.fromarray(bmp_array).save('images/' + name + '.bmp')


def input_gaussian(beam_type=0, intensity=1.0, beam_size=np.array((0.5, 0.5)), pos=np.array((0, 0))):
    if beam_type == 0:
        xx = intensity * np.exp(-(np.linspace(-1, 1, size[1]) - pos[0])**2 / (2 * beam_size[0]**2))
        yy = intensity * np.exp(-(np.linspace(-1, 1, size[0]) - pos[1])**2 / (2 * beam_size[1]**2))
    else:
        xx = intensity * np.ones(size[1])
        yy = intensity * np.ones(size[0])

    intensity_profile = np.meshgrid(xx, yy)
    phase_profile = np.ones(size) * np.exp(2j * pi)

    return intensity_profile * phase_profile


if __name__ == '__main__':

    toBMP(half(), name='half')
    toBMP(quad(), name='quad')

    toBMP(add(half(), correction), name='half_corr')
    toBMP(add(quad(), correction), name='quad_corr')

