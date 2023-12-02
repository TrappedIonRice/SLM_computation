# import slmsuite.holography.algorithms as algorithms
# import slmsuite.holography.toolbox.phase

import numpy as np
# import cupy as cp
import scipy
import matplotlib.pyplot as plt
import matplotlib

# import WGS
import slm
# import profile
# from InversePhase import inverse_phase

# matplotlib.use('QtAgg')


class ArrayModulator(slm.SLM):

    def __init__(self, size=np.array((1024, 1272)), correction_path="images/413corrwithLUT.bmp", beams=1, beam_positions=None, beam_sizes=None):
        super(ArrayModulator, self).__init__(size, correction_path)
        """
            size: SLM size in pixels
            correction_path: file location of the correction pattern
            beams: number of beams
            beam_positions: position of each beam
            beam_sizes: size of each beam
        """
        # Note: all coordinates are normalized to [-1, 1], and phases are on a 2pi scale

        # By default, space beams equally along the x axis
        if beam_positions is None:
            beam_positions = [[-1 + (i + 0.5) * 2 / beams, 0] for i in range(beams)]

        # By default, each beam gets an equal portion of the SLM in the x direction, and the entire SLM in y
        if beam_sizes is None:
            beam_sizes = [[2 / beams, 2] for _ in range(beams)]

        # if len(beams.shape) == 1:
        # Array of beams, where each element is [[xpos, ypos], [xsize, ysize]]
        self.beams = np.array([[beam_positions[i], beam_sizes[i]] for i in range(beams)])

        # Initial SLM Phase map
        self.phase = np.zeros(shape=size)

    # Define a region of size shape around pos, returning the normalized coordinates of [xmin, ymin] and [xmax, ymax]
    def region(self, pos, shape):
        corners = [pos - shape / 2, pos + shape / 2]
        corners = [np.clip(corner, -1, 1) for corner in corners]
        # print(pos)
        # print(shape)
        # print(corners)
        return corners

    # Add a phase pattern to a region of size shape centered around pos
    def region_add(self, pos, shape, phase):
        corners = [self.coord(corner) for corner in self.region(pos, shape)]
        # print(corners)
        self.phase[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]] = self.add(self.phase[corners[0][1]:corners[1][1], corners[0][0]:corners[1][0]], phase)

    # Apply the correction pattern to the entire SLM
    def correction(self, active_beams=None):
        if active_beams is None:
            active_beams = [i for i in range(len(self.beams))]

        self.phase = self.add(self.phase, self.correction)

    # Generate a TEM01 phase pattern of specified shape along specified axis
    def tem01(self, shape, axis=0):
        phase = np.transpose(np.zeros(shape=self.shape_px(shape)))
        if axis <= 0:
            phase[:, phase.shape[1] // 2:] = np.pi
        else:
            phase[phase.shape[0] // 2:, :] = np.pi
        return phase

    # Generate a flat phase shift
    def flat(self, shape, shift):
        phase = np.transpose(np.ones(shape=self.shape_px(shape))) * shift

        return phase

    # Generate a phase gradient to deflect incident light at angle (in radians)
    def gradient(self, shape, angle, axis=0):
        phase = np.transpose(np.zeros(shape=self.shape_px(shape)))
        delta = 2 * np.pi * self.pitch / 413 * np.tan(angle)
        if axis <= 0:
            for i in range(len(phase)):
                phase[i, :] += i * delta
        else:
            for i in range(len(phase[0])):
                phase[:, i] += i * delta

        return phase

    def rand(self, shape):
        phase = np.transpose(np.random.rand(self.shape_px(shape)[0], self.shape_px(shape)[1])) * 2 * np.pi
        return phase

    # Apply a phase shift of phase to each beam
    def add_phase(self, phase, active_beams=None):
        if active_beams is None:
            active_beams = [i for i in range(len(self.beams))]

        for beam in active_beams:
            self.region_add(self.beams[beam][0], self.beams[beam][1], phase)


if __name__ == '__main__':
    mod = ArrayModulator(beams=1)
    tem01 = mod.tem01(mod.beams[0][1], axis=0)
    # shift = mod.flat(mod.beams[0][1], np.pi / 8)
    # rand = mod.rand(mod.beams[0][1])
    grad = mod.gradient(mod.beams[0][1], angle=0.012 * np.pi / 180, axis=0)
    mod.add_phase(tem01)
    # mod.add_phase(shift, active_beams=[1])
    mod.add_phase(grad)
    # mod.add_phase(rand, active_beams=[0])
    # print(mod.phase)
    mod.phaseToBMP(mod.phase, name='slm_deflection_tem01_413', color=True, correction=True)
