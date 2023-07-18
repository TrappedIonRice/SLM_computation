import time

import matplotlib.pyplot
import numpy as np
from profile import Profile
from slm import SLM
from matplotlib import pyplot as plt
import scipy.stats

pi = np.pi


class WGS:

    def __init__(self, size=np.array((1024, 1272)), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 wavelength=413e-9, f=100):
        self.size = size

        self.input = input
        self.slm_field = input * np.exp(2j * pi * np.random.random_sample(size))
        self.A = np.abs(self.slm_field)
        self.B = np.zeros(size)
        self.phi = np.angle(self.slm_field)
        self.psi = np.zeros(size)
        self.g = np.array([np.ones(self.B.shape)])

        self.h = np.array([np.ones(self.psi.shape)])

        self.target = target[0]
        self.image_field = np.array(np.zeros(size))
        self.spots = target[1]

        self.wavelength = wavelength
        self.f = f

    def propagate(self):
        t = time.time()
        self.image_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.slm_field), norm="ortho"))
        print('FFT time:' + str(time.time() - t))

        self.B = np.abs(self.image_field)
        self.psi = np.angle(self.image_field)

    def opt(self):
        # print(self.spots)
        # print(self.B)
        # print(self.g.shape)
        avg_B = np.sum([self.B[m[0], m[1]] for m in self.spots]) / len(self.spots)
        g = np.zeros(self.B.shape)
        for m in self.spots:
            g[m[0], m[1]] += (avg_B / self.B[m[0], m[1]]) * self.g[-1][m[0], m[1]]
        # g = np.sum([(avg_B / self.B[-1][m[0], m[1]])[m[0], m[1]] for m in self.spots]) * self.g[-1]
        self.g = np.append(self.g, [g], axis=0)


        # avg_psi = np.sum([self.psi[-1][m[0], m[1]] for m in self.spots]) / len(self.spots)
        # h = np.zeros(self.psi[-1].shape)
        # for m in self.spots:
        #     h[m[0], m[1]] += (avg_psi / self.psi[-1][m[0], m[1]]) * self.h[-1][m[0], m[1]]
        # self.h = np.append(self.h, [h], axis=0)

    def backpropagate(self):
        t = time.time()
        backprop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(self.g[-1] * self.target * np.exp(1j * self.psi)),
                                               norm="ortho"))
        print('IFFT time:' + str(time.time() - t))
        # backprop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(self.g[-1] * np.abs(self.target) * np.exp(1j * (np.angle(self.target) + self.h[-1]))),
        #                                        norm="ortho"))
        self.A = np.abs(backprop)
        self.phi = np.angle(backprop)

        self.slm_field = self.input * np.exp(1j * self.phi)

    def iterate(self, n):
        for _ in range(n):
            self.propagate()
            self.opt()
            self.backpropagate()
            print('iteration')
        return self.slm_field, self.image_field

    def save_pattern(self, name, slm, correction=False):
        slm.ampToBMP(np.abs(self.input), name=name + '_input_amp')
        slm.phaseToBMP(self.phi, name=name + '_input_phase', correction=correction, color=True)
        slm.ampToBMP(self.B, name=name + '_output_amp')
        slm.phaseToBMP(self.psi, name=name + '_output_phase', color=True)

    def avg(self, field, pos, radius):
        avg = 0
        pts = 0
        for x in range(int(-radius * self.size[0]), int(radius * self.size[0])):
            for y in range(int(-radius * self.size[0]), int(radius * self.size[0])):
                if ((x / self.size[0]) ** 2 + (y / self.size[0]) ** 2) < radius ** 2:
                    avg += field[x + pos[0]][y + pos[1]]
                    pts += 1
        return avg / pts

    def beams(self, field=None, spots=None, waist=0.01):
        if field is None:
            field = self.image_field
        if spots is None:
            spots = self.spots
        return np.array([self.avg(field, m, waist) for m in spots])

    def dev_phase(self, spots=None, waist=0.01):
        if spots is None:
            spots = self.spots
        phases = [np.angle(self.avg(self.image_field, m, waist)) for m in spots]
        return scipy.stats.circstd(phases, high=pi, low=-pi)

    def dev_amp(self, spots=None, waist=0.01):
        if spots is None:
            spots = self.spots
        amps = [np.abs(self.avg(self.image_field, m, waist)) for m in spots]
        return np.std(amps) / np.max(amps)


class CostOptimizer(WGS):

    def __init__(self, size=np.array((1024, 1272)), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 start_phase=np.zeros((1024, 1272)), wavelength=413e-9, f=100):
        super().__init__(size, input, target, wavelength, f)
        self.slm_field = input * np.exp(1j * start_phase)
        self.image_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.slm_field), norm="ortho"))
        self.cost = self.cost_fxn()

    def cost_fxn(self, field=None, spots=None, waist=0.01):
        if spots is None:
            spots = self.spots
        if field is None:
            field = self.image_field

        beams = self.beams(field=field, spots=spots, waist=waist)
        avg_I = np.mean(np.abs(beams)**2) / np.max(np.abs(beams)**2)
        design_I = np.ones(beams.shape)

        gamma = np.sum(np.abs(beams)**2 * design_I) / np.sum(design_I**2)

        sigma = np.sqrt(np.sum(np.abs(beams)**2 - gamma * design_I)**2 / len(beams))

        f = 0.5

        return -avg_I + f * sigma

    def opt(self):
        phi = np.angle(self.slm_field)
        px = np.array(np.random.rand(2) * self.size, dtype=np.uint)

        phi[px[0], px[1]], phi[px[0], px[1]] = np.random.rand() * 2 * pi
        slm_field = self.input * np.exp(1j * phi)
        image_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(slm_field), norm="ortho"))
        cost = self.cost_fxn(field=image_field)

        if cost < self.cost:
            self.slm_field = slm_field
            self.image_field = image_field
            self.cost = cost

        return cost


def tem01(slm, size=(0.05, 0.05)):
    wgs = WGS(input=Profile.input_gaussian(beam_type=0, beam_size=np.array(size)))
    wgs.phi[-1] = slm.half()
    wgs.slm_field = wgs.input * np.exp(1j * wgs.phi[-1])
    wgs.propagate()
    wgs.save_pattern('TEM01', slm, correction=True)

    return wgs


def array2D(slm, n=4, m=4, x_pitch=0.02, y_pitch=0.02, size=(0.05, 0.05)):
    wgs = WGS(input=Profile.input_gaussian(beam_type=0, beam_size=np.array(size)),
              target=Profile.spot_array(m, n, x_pitch=y_pitch, y_pitch=x_pitch))
    wgs.iterate(20)
    wgs.save_pattern('%dx%d' % (n, m), slm)

    return wgs


def array1D(slm, n=5, pitch=0.02, size=(0.1, 0.1)):
    wgs = []
    min = 0
    for i in range(1):
        wgs.append(WGS(input=Profile.input_gaussian(beam_size=np.array(size)),
                       target=Profile.spot_array(1, n, y_pitch=pitch, center=(0.05, 0))))
        wgs[-1].iterate(20)
        if wgs[min].dev_phase(waist=0.002) > wgs[-1].dev_phase(waist=0.002):
            min = i
    wgs[min].save_pattern('1x%d' % n, slm, correction=True)

    print('Phase deviation: %.4f*Pi' % (wgs[min].dev_phase(waist=0.002) / (2 * pi)))
    print('Amplitude deviation: %.4f' % wgs[min].dev_amp(waist=0.002))

    return wgs


def tem01_2D(slm, n=1, m=5, wgs=None, size=(0.05, 0.05)):
    if wgs is None:
        wgs = array2D(slm, n, m)

    wgs2 = WGS(input=Profile.input_gaussian(beam_type=0, beam_size=np.array(size)))
    wgs2.phi[-1] = slm.add(wgs.phi[-1], slm.half())
    wgs2.slm_field = wgs2.input * np.exp(1j * wgs2.phi[-1])
    wgs2.propagate()
    wgs2.save_pattern('%dx%dTEM01' % (n, m), slm)
    slm.phaseToBMP(wgs2.phi[-1], name='%dx%dTEM01_input_phase' % (n, m), correction=True)


def interfere_wgs(slm):
    array1x5 = array1D(slm, 5, pitch=0.1)
    reference = Profile(field=Profile.input_gaussian(beam_type=0))
    interference = Profile(field=reference.field + array1x5.image_field)
    reference.save(slm, 'reference')
    interference.save(slm, 'interference_wgs')
    return interference.field


def interfere_ref(slm):
    array1x5 = Profile(field=Profile.gaussian_array(1, 5)[0]).field
    reference = Profile(field=Profile.input_gaussian(beam_type=0))
    interference = Profile(field=reference.field + array1x5)
    reference.save(slm, 'reference')
    interference.save(slm, 'interference_ref')
    return interference.field


if __name__ == '__main__':
    slm = SLM()

    # tem01(slm, size=(0.02, 0.02))

    # array4x4 = array2D(slm, 4, 4, x_pitch=0.08, y_pitch=0.08, size=(0.05, 0.05))
    array1x5 = array1D(slm, 5, pitch=0.004, size=(0.5, 0.5))
    #
    # tem01_2D(slm, 1, 5, array1x5)
    # tem01_2D(slm, 4, 4, array4x4)
    # interfere_ref(slm)


    # size = (0.05, 0.05)
    # reference = Profile(Profile.input_gaussian(beam_type=0, beam_size=np.array(size)) +
    #                     Profile.input_gaussian(beam_type=0, beam_size=np.array(size)))
    # interference = Profile(field=reference.field + array1x5.image_field)

    # interfere_wgs(slm)
