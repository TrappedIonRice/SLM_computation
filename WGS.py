import time

import matplotlib.pyplot
import numpy as np
from profile import Profile
from slm import SLM
from matplotlib import pyplot as plt
from InversePhase import inverse_phase
import scipy.stats
import sys

pi = np.pi


class IFTA:
    def __init__(self, size=np.array((1024, 1272)), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 wavelength=413e-9, f=100, waist=0.01):
        self.size = size
        self.waist = waist

        self.input = input
        self.slm_field = input * np.exp(2j * pi * np.random.random_sample(size))

        self.A = np.abs(self.slm_field)
        self.B = np.zeros(size)
        self.phi = np.angle(self.slm_field)
        self.psi = np.zeros(size)

        self.it = []
        self.B_dev = []
        self.psi_dev = []
        self.min_dev = (0, self.A, self.phi, self.B, self.psi)

        self.target = target[0]
        self.image_field = np.array(np.zeros(size))
        self.spots = target[1]

        self.wavelength = wavelength
        self.f = f

    def propagate(self, slm_field=None):
        if slm_field is None:
            slm_field = self.slm_field
        # t = time.time()
        self.image_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(slm_field), norm="ortho"))
        # print('FFT time:' + str(time.time() - t))

        self.B = np.abs(self.image_field)
        self.psi = np.angle(self.image_field)
        return self.image_field

    def opt(self):
        pass

    def backpropagate(self, image_field=None):
        if image_field is None:
            image_field = self.image_field

        # t = time.time()
        backprop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(image_field), norm="ortho"))
        # print('IFFT time:' + str(time.time() - t))
        self.A = np.abs(backprop)
        self.phi = np.angle(backprop)

        self.slm_field = self.input * np.exp(1j * self.phi)

    def iterate(self, n):
        for i in range(n):
            self.propagate()

            self.it.append(i + 1)
            self.psi_dev.append(self.dev_phase(waist=self.waist) / (2 * pi))
            self.B_dev.append(self.dev_amp(waist=self.waist))

            if len(self.B_dev) <= 1 or self.B_dev[-1] < np.min(self.B_dev[:-1]):
                self.min_dev = (i + 1, np.copy(self.A), np.copy(self.phi), np.copy(self.B), np.copy(self.psi),
                                    self.B_dev[-1], self.psi_dev[-1])

            self.opt()
            self.backpropagate()

            print('Step %d' % i)
        return self.slm_field, self.image_field

    def save_pattern(self, name, slm, correction=False, plots=(0, 1, 2, 3, 4, 5), min=False, target=True):
        if min:
            slm.ampToBMP(np.abs(self.input), name=name + '_input_amp', color=(0 in plots))
            slm.phaseToBMP(self.min_dev[2], name=name + '_input_phase', correction=correction, color=(1 in plots))

            slm.ampToBMP(self.min_dev[3], name=name + '_output_amp', color=(2 in plots))
            slm.phaseToBMP(self.min_dev[4], name=name + '_output_phase', color=(3 in plots))
        else:
            slm.ampToBMP(np.abs(self.slm_field), name=name + '_input_amp', color=(0 in plots))
            slm.phaseToBMP(np.angle(self.slm_field), name=name + '_input_phase', correction=correction, color=(1 in plots))

            slm.ampToBMP(np.abs(self.image_field), name=name + '_output_amp', color=(2 in plots))
            slm.phaseToBMP(np.angle(self.image_field), name=name + '_output_phase', color=(3 in plots))

        if target:
            slm.ampToBMP(np.abs(self.target), name=name + '_target_amp', color=(4 in plots))
            slm.phaseToBMP(np.angle(self.target), name=name + '_target_phase', color=(5 in plots))

    def avg(self, field, pos, radius):
        avg = 0
        pts = 0
        for x in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
            for y in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
                if ((x / self.size[0]) ** 2 + (y / self.size[0]) ** 2) <= radius ** 2:
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
        phases = [(np.angle(self.avg(self.image_field, m, waist)) + 2 * pi) % (2 * pi) for m in spots]
        return np.max([np.max([min(np.abs(p1 - p2), 2 * pi - np.abs(p1 - p2)) for p2 in phases]) for p1 in phases])
        # return scipy.stats.circstd(phases, high=pi, low=-pi)

    def dev_amp(self, spots=None, waist=0.01):
        if spots is None:
            spots = self.spots
        amps = [np.abs(self.avg(np.abs(self.image_field), m, waist)) for m in spots]
        return (np.max(amps) - np.min(amps)) / np.max(amps)

    def dev_intensity(self, spots=None, waist=0.01):
        if spots is None:
            spots = self.spots
        intensities = [self.avg(np.abs(self.image_field)**2, m, waist) for m in spots]
        return (np.max(intensities) - np.min(intensities)) / np.max(intensities)


class WGS(IFTA):

    def __init__(self, size=np.array((1024, 1272)), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 wavelength=413e-9, f=100, start_phase=None, reference=None, consider_phase=False, waist=0.01):
        super().__init__(size, input, target, wavelength, f, waist)
        self.consider_phase = consider_phase

        self.input = input
        if start_phase is not None:
            self.slm_field = input * np.exp(1j * pi * start_phase)
            self.A = np.abs(self.slm_field)
            self.phi = np.angle(self.slm_field)
        if reference is None:
            self.reference = np.zeros(size)
        else:
            self.reference = reference

        self.g = np.array([np.ones(self.B.shape)])

        self.h = np.array([np.ones(self.psi.shape)])

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


        if self.consider_phase:
            avg_psi = np.sum([self.psi[m[0], m[1]] for m in self.spots]) / len(self.spots)
            h = np.zeros(self.psi.shape)
            for m in self.spots:
                h[m[0], m[1]] += (avg_psi / self.psi[m[0], m[1]]) * self.h[-1][m[0], m[1]]
            self.h = np.append(self.h, [h], axis=0)

    def backpropagate(self, image_field=None):
        # t = time.time()
        if not self.consider_phase:
            backprop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(self.g[-1] * self.target * np.exp(1j * self.psi)),
                                                   norm="ortho"))
        else:
            backprop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(self.g[-1] * self.target * np.exp(1j * (self.psi + self.h[-1]))),
                                                   norm="ortho"))
        # print('IFFT time:' + str(time.time() - t))
        self.A = np.abs(backprop)
        self.phi = np.angle(backprop)

        self.slm_field = self.input * np.exp(1j * self.phi)

    def iterate(self, n):
        for i in range(n):
            self.propagate()

            self.it.append(i + 1)
            self.psi_dev.append(self.dev_phase(waist=self.waist) / pi)
            self.B_dev.append(self.dev_amp(waist=self.waist))

            if len(self.B_dev) <= 1 or self.B_dev[-1] < np.min(self.B_dev[:-1]):
                self.min_dev = (i + 1, np.copy(self.A), np.copy(self.phi), np.copy(self.B), np.copy(self.psi),
                                    self.B_dev[-1], self.psi_dev[-1])

            self.opt()
            self.backpropagate()

            print('Step %d' % i)
        return self.slm_field, self.image_field


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

    def gradient(self, phase_1D, h=0.01):
        phase = np.reshape(phase_1D, (-1, self.size[1]))

        current = self.cost_fxn(field=self.propagate(slm_field=phase))

        grad = np.zeros(phase_1D.shape)

        for i in range(len(phase)):
            for j in range(len(phase[i])):
                phase[i, j] += h
                grad[i * len(phase) + j] = (self.cost_fxn(field=self.propagate(slm_field=phase)) - current) / h
                phase[i, j] -= h


class OutputOutput(IFTA):
    def __init__(self, size=np.array((1024, 1272)), input=Profile.input_gaussian(), target=Profile.gaussian_array(1, 5),
                 wavelength=413e-9, f=100, waist=0.01, beta=1):
        super().__init__(size, input, target, wavelength, f, waist)
        self.propagate()

        self.a = np.copy(self.slm_field)
        self.b = np.copy(self.image_field)
        self.a_prime = []
        self.b_prime = []

        self.b_driving = []

        self.beta = beta

    def propagate(self, slm_field=None):
        if slm_field is None:
            slm_field = self.slm_field

        # t = time.time()
        self.image_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(slm_field), norm="ortho"))
        # print('FFT time:' + str(time.time() - t))

        self.B = np.abs(self.image_field)
        self.psi = np.angle(self.image_field)
        return self.image_field

    def opt(self):
        self.b_driving = np.abs(self.target) * (2 * np.exp(1j * np.angle(self.b_prime)) -
                                                     np.exp(1j * np.angle(self.b))) - self.b

        return self.b_prime + self.beta * self.b_driving

    def constraints(self, slm_field):
        if slm_field is None:
            slm_field = self.slm_field

        return np.abs(self.input) * np.angle(slm_field)

    def backpropagate(self, image_field=None):
        if image_field is None:
            image_field = self.image_field

        # t = time.time()
        backprop = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(image_field), norm="ortho"))
        # print('IFFT time:' + str(time.time() - t))
        self.A = np.abs(backprop)
        self.phi = np.angle(backprop)

        self.slm_field = self.input * np.exp(1j * self.phi)
        return self.slm_field

    def iterate(self, n):
        for i in range(n):
            self.a = self.backpropagate(image_field=self.b)
            self.a_prime = self.constraints(slm_field=self.a)
            self.b_prime = self.propagate(slm_field=self.a_prime)
            self.b = self.opt()

            self.slm_field = self.a_prime
            self.A = np.abs(self.slm_field)
            self.phi = np.angle(self.slm_field)

            self.image_field = self.b_prime
            self.B = np.abs(self.image_field)
            self.psi = np.angle(self.image_field)

            self.it.append(i + 1)
            self.psi_dev.append(self.dev_phase(waist=self.waist) / (2 * pi))
            self.B_dev.append(self.dev_amp(waist=self.waist))

            if len(self.B_dev) <= 1 or self.B_dev[-1] < np.min(self.B_dev[:-1]):
                self.min_dev = (i + 1, np.copy(self.A), np.copy(self.phi), np.copy(self.B), np.copy(self.psi),
                                self.B_dev[-1], self.psi_dev[-1])

            print('Step %d' % i)
        return self.slm_field, self.image_field


class ThreeStep(IFTA):
    def __init__(self):
        super().__init__()


class Wu(IFTA):
    def __init__(self, size=np.array((1024, 1272)), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 wavelength=413e-9, f=100, waist=0.01):
        super().__init__(size, input, target, wavelength, f, waist)

        self.p = 2 * pi * np.random.random_sample(size)

        self.image_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.input * np.exp(1j * self.p)), norm="ortho"))

        self.S = np.zeros(size)
        self.S[:, self.S.shape[1] // 2:] = 1

        # self.I = np.eye(self.size[0], M=self.size[1], k=0)
        self.I = np.ones(size)

        self.A_t = np.abs(self.target)
        self.P_t = np.angle(self.target)

        self.mask = np.where(np.abs(self.target) > 1e-3, 1, 0)
        # slm.ampToBMP(np.abs(self.mask), name='mask', color=True)

        self.eff = []
        self.nonunif = []
        self.phase_err = []

    def step(self):
        U_c = self.image_field
        A_c = np.abs(U_c)
        P_c = np.angle(U_c)

        A_alpha = self.A_t * self.S + A_c * (self.I - self.S)
        P_alpha = self.P_t * self.S + P_c * (self.I - self.S)
        U_alpha = A_alpha * np.exp(1j * P_alpha)
        u_alpha = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(U_alpha), norm="ortho"))
        p_alpha = np.angle(u_alpha)

        A_beta = self.A_t * (self.I - self.S) + A_c * self.S
        P_beta = self.P_t * (self.I - self.S) + P_c * self.S
        U_beta = A_beta * np.exp(1j * P_beta)
        u_beta = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(U_beta), norm="ortho"))
        p_beta = np.angle(u_beta)

        self.p = np.angle(np.exp(1j * p_alpha) + np.exp(1j * p_beta))
        self.slm_field = self.input * np.exp(1j * self.p)
        self.image_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.input * np.exp(1j * self.p)), norm="ortho"))

        self.image_field /= np.sqrt(np.sum(np.abs(self.image_field)**2))

    def iterate(self, N):
        for n in range(N):
            print(str(n) + ' ', end='')
            self.step()
            self.eff.append(self.eta())
            self.nonunif.append(self.dev_amp(waist=0.001))
            self.phase_err.append(self.phase_error())

        # self.image_field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.input * np.exp(1j * self.p)), norm="ortho"))

    def eta(self):
        return np.sum(np.abs(self.mask * self.image_field)**2) / np.sum(np.abs(self.image_field)**2)

    def I_a(self):
        return np.sum(self.mask * np.abs(self.image_field)**2) / np.sum(self.mask)

    def nonuniformity(self):
        # I_a = self.I_a()
        # I_a = np.sum(self.mask * np.abs(self.image_field)**2) / np.sum(self.mask)
        # I_t = np.sum(self.mask * np.abs(self.target)**2) / np.sum(self.mask)
        # normalized_image = self.image_field * np.sum(np.abs(self.input)) / np.sum(np.abs(self.image_field))
        return np.sum(self.mask * np.abs(np.abs(self.image_field) - np.abs(self.target))**2)\
               / np.sum(self.mask * np.abs(self.target)**2)

    def spot_nonuniformity(self):
        return

    def phase_error(self):
        return np.sum(self.mask * np.abs(self.image_field)**2 * np.abs(np.angle(self.image_field) - np.angle(self.target)))\
               / np.sum(self.mask * np.abs(self.image_field)**2 * np.pi)

# def dev_phase(image_field, spots, waist=0.01):
#     phases = [np.angle(self.avg(self.image_field, m, waist)) for m in spots]
#     return np.max([np.max([(p1 - p2) % (2 * pi) for p2 in phases]) for p1 in phases])
#     # return scipy.stats.circstd(phases, high=pi, low=-pi)
#
# def dev_amp(spots=None, waist=0.01):
#     if spots is None:
#         spots = self.spots
#     amps = [np.abs(self.avg(self.image_field, m, waist)) for m in spots]
#     return (np.max(amps) - np.min(amps)) / (np.max(amps) + np.min(amps))


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


def array1D(slm, it=20, tries=1, n=5, pitch=0.02, size=(0.05, 0.05), start=None, ref=None, consider_phase=False, waist=0.01, plots=(0, 1, 2, 3), add_noise=False):
    wgs = []
    min = 0
    start = [start]
    for i in range(tries):
        if add_noise:
            start.append(start[0] + (np.random.random_sample(slm.size) - 0.5) * 0.1)
        wgs.append(WGS(input=Profile.input_gaussian(beam_size=np.array(size)),
                       target=Profile.spot_array(1, n, y_pitch=pitch, center=(0.05, 0)), start_phase=start[-1], reference=None, consider_phase=consider_phase, waist=waist))
        wgs[-1].iterate(it)
        if wgs[min].min_dev[6] > wgs[-1].min_dev[6]:
            min = i
    wgs[min].save_pattern('1x%d' % n, slm, correction=True, plots=plots, min=True)
    wgs[min].A, wgs[min].phi, wgs[min].B, wgs[min].psi = wgs[min].min_dev[1:5]
    wgs[min].slm_field = wgs[min].A * np.exp(wgs[min].phi * 1j)
    wgs[min].image_field = wgs[min].B * np.exp(wgs[min].psi * 1j)

    print('Minimum amplitude non-uniformity at iteration %d out of %d' % (wgs[min].min_dev[0], it))
    print('Min Amplitude non-uniformity: %f' % wgs[min].min_dev[5])
    print('Phase non-uniformity: %f*Pi' % wgs[min].min_dev[6])
    print('Phases in last iteration: ' + str([np.angle(wgs[min].avg(wgs[min].image_field, m, waist)) / pi for m in wgs[min].spots]) + ' * pi')

    plt.figure(1)
    plt.clf()
    plt.plot(wgs[min].it, wgs[min].B_dev, label='Amplitude non-uniformity')
    plt.plot(wgs[min].it, wgs[min].psi_dev, label='Phase non-uniformity')
    plt.title('Convergence')
    plt.ylabel('Non-uniformity')
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.legend()
    plt.show()

    return wgs[min]


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


def array1D_OutputOutput(slm, it=20, n=5, pitch=0.02, size=(0.05, 0.05), start=None, ref=None, waist=0.01, plots=(0, 1, 2, 3)):

    oo = OutputOutput(input=Profile.input_gaussian(beam_size=np.array(size)),
                      target=Profile.gaussian_array(1, 5, x_pitch=pitch, waist=(waist, waist)), waist=waist)

    print(oo.spots)

    oo.iterate(it)

    oo.save_pattern('1x%d' % n, slm, correction=True, plots=plots, min=True)

    print('Minimum amplitude non-uniformity at iteration %d out of %d' % (oo.min_dev[0], it))
    print('Min Amplitude non-uniformity: %f' % oo.min_dev[5])
    print('Phase non-uniformity: %f*Pi' % oo.min_dev[6])

    plt.figure(1)
    plt.clf()
    plt.plot(oo.it, oo.B_dev, label='Amplitude non-uniformity')
    plt.plot(oo.it, oo.psi_dev, label='Phase non-uniformity')
    plt.title('Image Convergence')
    plt.ylabel('Non-uniformity')
    plt.xlabel('Elapsed iterations')
    plt.yscale('log')
    plt.legend()
    plt.show()

    return oo


def wu(slm, N=40, M=10, n=5, plot_each=False):

    input_profile = Profile.input_gaussian(beam_size=(0.5, 0.5))
    input_profile /= np.sqrt(np.sum(np.abs(input_profile)**2))

    amps = [1 for _ in range(n)]
    phases = [0 for _ in range(n)]

    eff = []
    nonunif = []
    phase_err = []

    wus = []
    slm_fields = []
    image_fields = []

    for i in range(M):
        print('Iteration: ' + str(M))
        target = Profile.target_output_array(1, n, input_profile=input_profile, x_pitch=0.008, amps=amps, phases=phases)

        # print(np.sum(np.abs(input_profile)**2))
        # print(np.sum(np.abs(target[0])**2))
        # target = Profile.gaussian_array(1, 5, waist=(0.02, 0.02))
        # target[0] *= np.exp(1j * np.pi / 2)

        wu = Wu(input=input_profile, target=target)
        # print(wu.target.shape)
        wu.iterate(N)
        # print(np.sum(np.abs(wu.image_field)**2))
        # temp = wu.image_field
        # wu.image_field = np.abs(wu.image_field) * np.exp(1j * np.angle(wu.image_field) * wu.mask)
        if plot_each:
            wu.save_pattern('wu_1x5', slm, target=False)
        result = wu.beams(waist=0.001)
        amps /= np.abs(result)**0.5
        amps /= np.max(amps)
        print(amps)
        # phases = -1 * np.angle(result)
        # wu.image_field = temp
        eff.append(wu.eta())
        nonunif.append(wu.dev_amp(waist=0.001))
        phase_err.append(wu.phase_error())

        wus.append(wu)

        print('Diffraction Efficiency: ' + str(eff[-1]))
        print('Amplitude nonuniformity: ' + str(nonunif[-1]))
        print('Phase error: ' + str(phase_err[-1]))

        # print(wu.beams(waist=0.005))

        # print(wu.I_a())

        if plot_each:
            plt.plot([n for n in range(N)], wu.eff, label='Efficiency')
            plt.plot([n for n in range(N)], wu.nonunif, label='Amplitude Nonuniformity')
            plt.plot([n for n in range(N)], wu.phase_err, label='Phase error')
            plt.xlabel('Inner Iteration')
            plt.xlim(0, N)
            # plt.ylim(0, 1)
            plt.grid(True)
            plt.legend()

            plt.show()

    min_it = nonunif.index(min(nonunif))

    wus[min_it].save_pattern(name='wu_1x5', slm=slm, target=False, correction=True)

    print('')
    print('----Minimum iteration----')
    print('Diffraction Efficiency: ' + str(eff[min_it]))
    print('Amplitude nonuniformity: ' + str(nonunif[min_it]))
    print('Phase error: ' + str(phase_err[min_it]))

    plt.plot([n for n in range(N)], wus[min_it].eff, label='Efficiency')
    plt.plot([n for n in range(N)], wus[min_it].nonunif, label='Amplitude Nonuniformity')
    plt.plot([n for n in range(N)], wus[min_it].phase_err, label='Phase error')
    plt.xlabel('Inner Iteration')
    plt.xlim(0, N)
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot([m for m in range(M)], eff, label='Efficiency')
    plt.plot([m for m in range(M)], nonunif, label='Amplitude Nonuniformity')
    plt.plot([m for m in range(M)], phase_err, label='Phase error')
    plt.xlabel('Outer Iteration')
    plt.xlim(0, M)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    slm = SLM()

    # tem01(slm, size=(0.02, 0.02))
    input_size = (0.05, 0.05)

    # target, spots = Profile.gaussian_array(1, 5, amps=np.exp(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * np.pi * 1j), y_pitch=0.1, waist=(0.05, 0.05))
    #
    # # array4x4 = array2D(slm, 4, 4, x_pitch=0.08, y_pitch=0.08, size=(0.05, 0.05))
    # # start_phase = np.random.random_sample(slm.size)
    # start_phase = inverse_phase(slm, color=True, target=target, spots=spots, input_size=input_size)
    # # start_phase = start_phase + np.random.random_sample(slm.size) * 0.5
    # # start_phase = (start_phase + slm.half())
    # slm.phaseToBMP(start_phase, '1x5_start_phase', color=True)
    # array1x5 = array1D(slm, it=20, tries=1, n=5, pitch=0.1, size=input_size, consider_phase=False, plots=(2, 3), start=start_phase, add_noise=False, waist=0.05)
    #
    # phases = -1 * np.angle(array1x5.beams())/2
    # updated_amps = np.exp(1j * phases)
    # updated_target, spots = Profile.gaussian_array(1, 5, amps=updated_amps)
    #
    # start_phase = inverse_phase(slm, color=True, target=updated_target, spots=spots, input_size=input_size)
    # array1x5 = array1D(slm, it=20, tries=1, n=5, pitch=0.1, size=input_size, consider_phase=False, plots=(2, 3),
    #                    start=start_phase, add_noise=False, waist=0.05)




    # start_phase = slm.BMPToPhase('images/blaze_grating.bmp')
    # ifta = IFTA(input=Profile.input_gaussian(beam_size=input_size))
    # ifta.phi = start_phase
    # ifta.slm_field = np.abs(ifta.slm_field) * np.exp(1j * ifta.phi)
    # ifta.propagate()
    # ifta.save_pattern('blaze_grat',slm)




    # array1x5_2 = array1D(slm, it=30, n=5, pitch=0.1, size=(0.05, 0.05), start=array1x5.phi, consider_phase=True, plots=(2, 3))

    #
    # tem01_2D(slm, 1, 5, array1x5)
    # tem01_2D(slm, 4, 4, array4x4)
    # interfere_ref(slm)


    # size = (0.05, 0.05)
    # reference = Profile(Profile.input_gaussian(beam_type=0, beam_size=np.array(size)) +
    #                     Profile.input_gaussian(beam_type=0, beam_size=np.array(size)))
    # interference = Profile(field=reference.field + array1x5.image_field)

    # interfere_wgs(slm)

    # array1x5_oo = array1D_OutputOutput(slm, it=100, n=5, pitch=0.1, size=(0.05, 0.05), waist=0.05, plots=(2, 3))

    wu(slm, 40)

    # ifta = IFTA(input=Profile.input_gaussian(beam_size=(0.05, 0.05)))
    # ifta.slm_field = np.abs(ifta.slm_field)
    # ifta.propagate(ifta.slm_field)
    # ifta.backpropagate(ifta.image_field)
    # ifta.save_pattern('test', slm)
