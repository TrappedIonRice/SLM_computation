from PIL import Image as im
import numpy as np

pi = np.pi


# All coordinates and amplitudes normalized to [-1, 1]
class Profile:

    def __init__(self, size=np.array((1024, 1272)), field=None):

        if field is None:
            field = Profile.input_gaussian(size=size)
        self.field = field

        self.amp = np.abs(self.field)
        self.phase = np.angle(self.field)

        self.size = self.field.shape

    def save(self, slm, name):
        slm.ampToBMP(self.amp, name=name + '_amp')
        slm.phaseToBMP(self.phase, name=name + '_phase', color=True)

    @staticmethod
    def input_gaussian(beam_type=0, amp=1.0, beam_size=np.array((0.5, 0.5)), pos=np.array((0, 0)),
                       size=np.array((1024, 1272))):

        if beam_type == 0:
            # beam_size[0] *= size[0] / size[1]
            beam_size = np.array([beam_size[0] * size[0] / size[1], beam_size[1]])
            xx = amp * np.exp(-(np.linspace(-1, 1, size[1]) - pos[0])**2 / beam_size[0]**2)
            yy = amp * np.exp(-(np.linspace(-1, 1, size[0]) - pos[1])**2 / beam_size[1]**2)
        else:
            xx = amp * np.ones(size[1])
            yy = amp * np.ones(size[0])

        beams = np.meshgrid(xx, yy)
        amp_profile = beams[0] * beams[1]
        phase_profile = np.ones(size) * np.exp(2j * pi)

        return amp_profile * phase_profile

    @staticmethod
    def spot_array(n, m, center=np.array((0, 0)), x_pitch=0.1, y_pitch=0.1, size=np.array((1024, 1272))):
        amp_profile = np.zeros(size)
        spots = np.array([[0, 0]])
        for i in np.linspace(-0.5 * n * x_pitch + center[0], 0.5 * n * x_pitch + center[0], n):
            for j in np.linspace(-0.5 * m * y_pitch + center[1], 0.5 * m * y_pitch + center[1], m):
                # x = (i + 0.5) * size[0]
                amp_profile[int((i + 0.5) * size[0]), int((j + 0.5) * size[1])] = 1.0
                spots = np.append(spots, [[int((i + 0.5) * size[0]), int((j + 0.5) * size[1])]], axis=0)
        return [amp_profile * np.exp(2j * pi), spots[1:]]

    @staticmethod
    def gaussian_array(n, m, waist=(0.02, 0.02), center=np.array((0, 0)), x_pitch=0.1, y_pitch=0.1,
                       size=np.array((1024, 1272))):
        spot_array, spots = Profile.spot_array(m, n, center, x_pitch, y_pitch, size)
        amp = np.copy(spot_array)
        for spot in spots:
            amp += Profile.input_gaussian(beam_size=waist, pos=(spot / size - 0.5) * 2, size=size)

        amp = np.abs(amp)
        amp /= np.max(amp)
        return [amp, spots]

