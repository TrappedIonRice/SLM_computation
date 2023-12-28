from PIL import Image as im
import numpy as np
import slm

pi = np.pi


def shift_array(arr, shift=(0, 0)):
    shift = np.array(np.array(shift) * arr.shape, dtype=np.uint)
    # out = np.roll(arr, shift, axis=(0, 1))
    out = np.zeros(arr.shape)
    for i in range(len(arr)):
        for j in range(len(arr)):
            if i + shift[0] > 0 and j + shift[1] > 0:
                out[i, j] = arr[i + shift[0], j + shift[1]]
    return out


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

    # Generate single gaussian beam
    @staticmethod
    def input_gaussian(beam_type=0, amp=1.0, beam_size=np.array((0.5, 0.5)), pos=np.array((0, 0)),
                       size=np.array((1024, 1272)), mesh=False):

        if beam_type == 0:
            # beam_size[0] *= size[0] / size[1]
            beam_size = np.array([beam_size[0] * size[0] / size[1], beam_size[1]])
            xx = np.exp(-(np.linspace(-1, 1, size[1]) - pos[1])**2 / beam_size[0]**2)
            yy = np.exp(-(np.linspace(-1, 1, size[0]) - pos[0])**2 / beam_size[1]**2)
        else:
            xx = np.ones(size[1])
            yy = np.ones(size[0])

        beams = np.meshgrid(xx, yy)
        # print(amp)
        amp_profile = amp * beams[0] * beams[1]
        # phase_profile = np.ones(size) * np.exp(2j * pi)

        if mesh:
            return amp_profile, beams

        return amp_profile

    # Generate array of single pixel spots
    @staticmethod
    def spot_array(n, m, center=np.array((0, 0)), x_pitch=0.1, y_pitch=0.1, size=np.array((1024, 1272))):
        amp_profile = np.zeros(size)
        spots = np.array([[0, 0]])
        for i in np.linspace(-0.5 * n * x_pitch + center[0], 0.5 * n * x_pitch + center[0], n, endpoint=True):
            if n <= 1:
                i = 0
            if m > 1:
                for j in np.linspace(-0.5 * m * y_pitch + center[1], 0.5 * m * y_pitch + center[1], m, endpoint=True):
                    # x = (i + 0.5) * size[0]
                    amp_profile[int((i + 0.5) * size[0]), int((j + 0.5) * size[1])] = 1.0
                    spots = np.append(spots, [[int((i + 0.5) * size[0]), int((j + 0.5) * size[1])]], axis=0)
            else:
                amp_profile[int((i + 0.5) * size[0]), int(0.5 * size[1])] = 1.0
                spots = np.append(spots, [[int((i + 0.5) * size[0]), int(0.5 * size[1])]], axis=0)
        return [amp_profile * np.exp(2j * pi), spots[1:]]

    # Generate array of gaussian beams
    @staticmethod
    def gaussian_array(n, m, waist=(0.02, 0.02), center=np.array((0, 0)), x_pitch=0.1, y_pitch=0.1,
                       size=np.array((1024, 1272)), amps=None):

        if amps is None:
            amps = [1 for _ in range(n * m)]
        amps = np.array(amps)
        # print(amps)

        spot_array, spots = Profile.spot_array(n, m, center, x_pitch, y_pitch, size)
        amp = np.copy(spot_array)
        for i in range(len(spots)):
            amp += Profile.input_gaussian(beam_size=waist, pos=(spots[i] / size - 0.5) * 2, size=size, amp=amps[i])

        amp -= spot_array
        # amp = np.abs(amp)
        amp /= np.max(np.abs(amp))
        return [amp, spots]

    # Generate a target output array of gaussian beams
    @staticmethod
    def target_output_array(n, m, input_profile, center=np.array((0, 0)),
                              x_pitch=0.1, y_pitch=0.1, size=np.array((1024, 1272)), amps=None, phases=None,
                              global_phase=0):

        if amps is None:
            amps = [1 for _ in range(n * m)]
        amps = np.array(amps)

        if phases is None:
            phases = [global_phase for _ in range(n * m)]
        phases = np.array(phases)
        # print(amps)

        # input_profile = Profile.input_gaussian(beam_size=input_waist, pos=input_center, size=size, amp=1)
        transform = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(input_profile), norm="ortho"))

        spot_array, spots = Profile.spot_array(n, m, center, y_pitch, x_pitch, size)
        amp = np.zeros(transform.shape, dtype=np.complex128)
        for i in range(len(spots)):
            # shift = np.array(np.array(shift) * arr.shape, dtype=np.uint)
            shift = (int(spots[i][0] - size[0] / 2), int(spots[i][1] - size[1] / 2))

            amp += np.abs(np.roll(transform, shift, axis=(0, 1))) * amps[i] * np.exp(1j * phases[i])
            # amp += Profile.input_gaussian(beam_size=waist, pos=(spots[i] / size - 0.5) * 2, size=size, amp=amps[i])

            # slm.ampToBMP(np.abs(amp), 'shifted_transform', True)
        # amp -= spot_array
        # amp = np.abs(amp)
        # amp /= np.max(np.abs(amp))
        # total_input = np.sum(np.abs(input_profile))
        amp *= np.sqrt(np.sum(np.abs(input_profile)**2) / np.sum(np.abs(amp)**2))
        return [amp, spots]


if __name__ == '__main__':
    slm = slm.SLM()
    input_size = np.array((0.05, 0.05))
    input_profile = Profile.input_gaussian(beam_size=input_size, pos=np.array((0, 0)), size=slm.size, amp=1)

    target_array = Profile.target_output_array(1, 5, input_profile=input_profile)

    slm.ampToBMP(input_profile, 'input_profile', True)
    slm.ampToBMP(np.abs(target_array[0]), 'target_array', True)
    slm.phaseToBMP(np.angle(target_array[0]), 'target_array', color=True)
