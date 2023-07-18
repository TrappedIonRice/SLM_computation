import matplotlib.colors
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('TkAgg')

pi = np.pi


# All coordinates normalized to [-1, 1]
class SLM:

    def __init__(self, size=np.array((1024, 1272)), correction_path="images/413corrwithLUT.bmp", lut=None):
        if lut is None:
            lut = {413: 103}
        self.lut = lut

        self.size = size

        correction_image = im.open(correction_path)
        self.correction = np.array(correction_image) / lut[413] * 2 * pi

    def half(self, center=0):
        center = np.array(self.size * (center / 2 + 0.5), np.uint)
        phase = np.zeros(self.size)
        phase[:, :center[1]] = pi
        return phase

    def quad(self, center=np.array((0, 0))):
        center = np.array(self.size * (center / 2 + 0.5), np.uint)
        phase = np.zeros(self.size)
        phase[:center[0], :center[1]] = pi
        phase[center[0]:, center[1]:] = pi
        return phase

    def random(self):
        return np.random.rand(self.size)

    def add(self, a, b):
        out = (a + b) % (2 * pi)
        return out

    def cyclic_colormap(self, grayscale):
        # print(grayscale)
        hsv = np.append(np.expand_dims(grayscale, axis=2), np.ones((grayscale.shape[0], grayscale.shape[1], 2)), axis=2)
        # print(hsv)
        # rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return grayscale

    def phaseToBMP(self, phase, name='output', wavelength=413, correction=False, color=False):
        phase = (phase + 2 * pi) % (2 * pi)
        if correction:
            phase = self.add(phase, self.correction)

        if color:
            fig = plt.figure(dpi=240)
            plt.imshow(phase / (2 * pi), cmap='hsv')
            plt.colorbar()
            plt.xlabel('X (px)')
            plt.ylabel('Y (px)')
            plt.title(name)
            fig.tight_layout()
            plt.show()
            im.frombytes('RGB', fig.canvas.get_width_height(),
                         fig.canvas.tostring_rgb()).save('images/' + name + '_color.png')

        bmp_array = np.array(phase / (2 * pi) * self.lut[wavelength], dtype=np.uint8)
        im.fromarray(bmp_array).save('images/' + name + '.bmp')

    def ampToBMP(self, amp, name='output'):
        bmp_array = np.array(((amp / np.max(amp)) ** 2) * 255, dtype=np.uint8)
        im.fromarray(bmp_array).save('images/' + name + '.bmp')

        fig = plt.figure(dpi=240)
        plt.imshow(amp)
        plt.colorbar()
        plt.xlabel('X (px)')
        plt.ylabel('Y (px)')
        plt.title(name)
        fig.tight_layout()
        plt.show()
        im.frombytes('RGB', fig.canvas.get_width_height(),
                     fig.canvas.tostring_rgb()).save('images/' + name + '_color.png')

    def BMPToPhase(self, path):
        return np.array(im.open(path)) / self.lut[413] * 2 * pi

    def BMPToAmp(self, path):
        out = np.array(im.open(path))
        return out / np.max(out)


if __name__ == '__main__':
    slm = SLM()

    half = slm.half()
    quad = slm.quad()

    half_corr = slm.add(half, slm.correction)
    quad_corr = slm.add(quad, slm.correction)

    slm.phaseToBMP(half, name='half')
    slm.phaseToBMP(quad, name='quad')
    slm.phaseToBMP(half_corr, name='half_corr')
    slm.phaseToBMP(quad_corr, name='quad_corr')
