import matplotlib.colors
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('TkAgg')

pi = np.pi


# All coordinates normalized to [-1, 1]
# Class to represent an SLM
class SLM:

    def __init__(self, size=np.array((1024, 1272)), correction_path="images/413corrwithLUT.bmp", lut=None, pitch=12500):
        # LUT for number of phase increments on SLM pixels
        if lut is None:
            lut = {411: 102, 413: 103, 435:114}
        self.lut = lut

        self.size = np.array(size)
        self.pitch = pitch

        # Correction pattern to be applied for the respective wavelength
        correction_image = im.open(correction_path)
        self.correction = np.array(correction_image) / lut[413] * 2 * pi

    # Convert normalized [-1, 1] coordinates to array indices
    def coord(self, r=np.array((0, 0))):
        shape = np.array([self.size[1], self.size[0]])
        r_px = np.array(shape * (r / 2 + 0.5), np.uint)
        return r_px

    #
    def shape_px(self, shape):
        shape_px = np.array([self.size[1], self.size[0]])
        return np.array(shape_px * shape / 2, np.uint)

    # Return an otherwise flat phase pattern with a pi shift from one half to the other
    def half(self, center=0):
        center = np.array(self.size * (center / 2 + 0.5), np.uint)
        phase = np.zeros(self.size)
        phase[:, :center[1]] = pi
        return phase

    # Return a 2x2 checkerboard phase pattern (with pi shifts between squares)
    def quad(self, center=np.array((0, 0))):
        center = np.array(self.size * (center / 2 + 0.5), np.uint)
        phase = np.zeros(self.size)
        phase[:center[0], :center[1]] = pi
        phase[center[0]:, center[1]:] = pi
        return phase

    # Generate random phase pattern
    def random(self):
        return np.random.rand(self.size)

    # Add two phase patterns modulo 2pi
    def add(self, a, b):
        out = (a + b) % (2 * pi)
        return out

    # Convert grayscale phase pattern to a cyclic colormap for better visualization
    def cyclic_colormap(self, grayscale):
        # print(grayscale)
        hsv = np.append(np.expand_dims(grayscale, axis=2), np.ones((grayscale.shape[0], grayscale.shape[1], 2)), axis=2)
        # print(hsv)
        # rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return grayscale

    # Plot and save a phase pattern
    def phaseToBMP(self, phase, name='output', wavelength=413, correction=False, color=False, show=False, fig=None):
        phase = (phase + 2 * pi) % (2 * pi)
        if correction:
            phase = self.add(phase, self.correction)

        if color:
            fig = plt.figure(fig, dpi=150)
            plt.clf()
            # fig = plt.figure(dpi=150)
            phase[0, 0] = 0
            phase[-1, -1] = 2 * pi
            plt.imshow(phase / pi, cmap='hsv')
            plt.colorbar()
            plt.xlabel('X (px)')
            plt.ylabel('Y (px)')
            plt.title(name + ' (pi radians)')
            fig.tight_layout()
            plt.pause(.001)
            if show:
                plt.show()
                # plt.draw()
            im.frombytes('RGB', fig.canvas.get_width_height(),
                         fig.canvas.tostring_rgb()).save('images/' + name + '_color.png')

        bmp_array = np.array(phase / (2 * pi) * self.lut[wavelength], dtype=np.uint8)
        im.fromarray(bmp_array).save('images/' + name + '.bmp')

    # Plot and save an amplitude pattern
    def ampToBMP(self, amp, name='output', color=False, show=False, fig=None):
        bmp_array = np.array(((amp / np.max(amp)) ** 2) * 255, dtype=np.uint8)
        im.fromarray(bmp_array).save('images/' + name + '.bmp')

        if color:
            fig = plt.figure(fig, dpi=240)
            plt.clf()
            # fig = plt.figure(dpi=240)
            amp[0, 0] = 0
            plt.imshow(amp)
            plt.colorbar()
            plt.xlabel('X (px)')
            plt.ylabel('Y (px)')
            plt.title(name)
            fig.tight_layout()
            plt.pause(.001)
            if show:
                plt.show()
                # plt.draw()
            im.frombytes('RGB', fig.canvas.get_width_height(),
                         fig.canvas.tostring_rgb()).save('images/' + name + '_color.png')

    # Import a phase pattern from an image
    def BMPToPhase(self, path):
        return np.array(im.open(path)) / self.lut[413] * 2 * pi

    # Import an amplitude pattern from an image
    def BMPToAmp(self, path):
        out = np.array(im.open(path))
        return out / np.max(out)


if __name__ == '__main__':
    slm = SLM()

    # half = slm.half()
    # quad = slm.quad()
    #
    # half_corr = slm.add(half, slm.correction)
    # quad_corr = slm.add(quad, slm.correction)
    #
    # slm.phaseToBMP(half, name='half')
    # slm.phaseToBMP(quad, name='quad')
    # slm.phaseToBMP(half_corr, name='half_corr')
    # slm.phaseToBMP(quad_corr, name='quad_corr')

    wu_1x5_corr = slm.add(slm.BMPToPhase('images/wu_1x5_input_phase.bmp'), slm.correction)
    slm.phaseToBMP(wu_1x5_corr, name='wu_1x5_input_phase_corr')
