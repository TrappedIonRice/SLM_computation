import matplotlib.colors
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import colorsys
import skimage.color

# matplotlib.use('TkAgg')

pi = np.pi


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

# All coordinates normalized to [-1, 1]
# Class to represent an SLM
class SLM:

    def __init__(self, size=np.array((1024, 1272)), correction_path="images/413corrwithLUT.bmp", lut=None, pitch=12500,
                 wavelength=413):
        # LUT for number of phase increments on SLM pixels
        if lut is None:
            lut = {399: 93, 411: 102, 413: 103, 435:114}
        self.lut = lut

        self.size = np.array(size)
        self.pitch = pitch

        # Correction pattern to be applied for the respective wavelength
        correction_image = im.open(correction_path)
        self.correction = np.array(correction_image) / lut[wavelength] * 2 * pi

        # self.correction_399 = np.array(correction_image) / lut[399] * 2 * pi

    # Convert normalized [-1, 1] coordinates to array indices
    def coord(self, r=np.array((0, 0))):
        shape = np.array([self.size[1], self.size[0]])
        r_px = np.array(shape * (r / 2 + 0.5), np.uint)
        return r_px

    def px_to_coords(self, r_px=np.array((0, 0))):
        shape = np.array([self.size[1], self.size[0]])
        r = np.array(2 * (r_px / shape - 0.5))
        return r

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

    def pixelate(self, field, factor=2):
        out = np.zeros(field.shape / factor)
        for i in range(len(out)):
            for j in range(len(out)):
                out[i][j] = np.average(field[i * factor:(i + 1) * factor][j * factor:(j + 1) * factor])
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
            fig = plt.figure(fig, dpi=150, figsize=(5, 4))
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
            # return fig

        bmp_array = np.array(phase / (2 * pi) * self.lut[wavelength], dtype=np.uint8)
        im.fromarray(bmp_array).save('images/' + name + '.bmp')

    # Plot and save an amplitude pattern
    def ampToBMP(self, amp, name='output', color=False, show=False, fig=None):
        bmp_array = np.array(((amp / np.max(amp)) ** 2) * 255, dtype=np.uint8)
        im.fromarray(bmp_array).save('images/' + name + '.bmp')

        if color:
            fig = plt.figure(fig, dpi=240, figsize=(5, 4))
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
            return fig

        # Plot and save an amplitude pattern

    def fieldtoBMP(self, field, name='output', wavelength=413, correction=False, color=False, show=False, fig=None):
        field = field / np.max(np.abs(field))
        # hsv_array = np.array([np.abs(field)**2, np.ones(np.shape(field)), np.angle(field * 255 / (2 * np.pi))])
        # print('one')
        # print(matplotlib.colors.hsv_to_rgb([0.9, 0.9, 0.9]))
        bmp_array = np.array([((np.angle(field) + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi), np.ones(self.size), np.abs(field)**2])
        bmp_array = np.swapaxes(bmp_array, 0, 2)
        bmp_array = np.swapaxes(bmp_array, 0, 1)
        # print(np.shape(bmp_array))
        # bmp_array = np.swapaxes(bmp_array, 0, 2)
        # print(np.shape(bmp_array))
        bmp_array = np.array(skimage.color.convert_colorspace(bmp_array, 'HSV', 'RGB', channel_axis=-1) * 255, dtype=np.uint8)
        # bmp_array = np.array([np.transpose([[hsv2rgb(np.angle(col) / 2 / np.pi, 1, np.abs(col)**2)[i] for col in row] for row in field]) for i in range(3)], dtype=np.uint8)
        # print('two')
        # bmp_array = np.swapaxes(bmp_array, 0, 2)
        # print(np.shape(bmp_array))
        image = im.fromarray(bmp_array, mode='RGB')
        # print('three')
        image.save('images/' + name + '.bmp')
        # print('four')

        phase = np.angle(field)
        amp = np.abs(field)

        phase = (phase + 2 * pi) % (2 * pi)
        if correction:
            phase = self.add(phase, self.correction)
        phase_bmp_array = np.array(phase / (2 * pi) * self.lut[wavelength], dtype=np.uint8)
        im.fromarray(phase_bmp_array).save('images/' + name + '_phase.bmp')

        amp_bmp_array = np.array(((amp / np.max(amp)) ** 2) * 255, dtype=np.uint8)
        im.fromarray(amp_bmp_array).save('images/' + name + '_amp.bmp')

        if color:
            fig = plt.figure(fig, dpi=240, figsize=(5, 4))
            # plt.clf()
            # fig = plt.figure(dpi=240)
            field[0, 0] = 0
            plt.imshow(image)
            plt.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='hsv'), ax=fig.axes[0], label='$2\\pi$ radians')
            # plt.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=None, cmap='Greys'), ax=fig.axes[0], label='Intensity')
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
            return fig

    # Import a phase pattern from an image
    def BMPToPhase(self, path, wavelength=413):
        return np.array(im.open(path)) / self.lut[wavelength] * 2 * pi

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
