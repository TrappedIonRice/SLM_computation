import time

import matplotlib.pyplot
import numpy as np
from profile import Profile
from slm import SLM
from matplotlib import pyplot as plt
import cupy as cp
from InversePhase import inverse_phase
import scipy.stats
import sys

pi = cp.pi


# Wrapper class for iterative fourier transform algorithms
class IFTA:

    # Initialize an IFTA for an SLM of specified size, with specified input and target light fields
    def __init__(self, size=(1024, 1272), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 wavelength=413e-9, f=100, waist=0.01):
        self.size = size
        self.waist = waist

        # Initial SLM phase is random
        self.input = input
        self.slm_field = input * cp.exp(2j * pi * cp.random.random_sample(size))

        self.A = cp.abs(self.slm_field)
        self.B = cp.zeros(size)
        self.phi = cp.angle(self.slm_field)
        self.psi = cp.zeros(size)

        # Keep track of iterations and deviations from target phase and amplitude for each iteration
        self.it = []
        self.B_dev = []
        self.psi_dev = []
        self.min_dev = (0, self.A, self.phi, self.B, self.psi)

        self.target = target[0]
        self.image_field = cp.array(cp.zeros(size))
        self.spots = target[1]

        self.wavelength = wavelength
        self.f = f

    # Propagate the SLM plane light field to the image plane
    def propagate(self, slm_field=None):
        if slm_field is None:
            slm_field = self.slm_field
        # t = time.time()
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slm_field), norm="ortho"))
        # print('FFT time:' + str(time.time() - t))

        self.B = cp.abs(self.image_field)
        self.psi = cp.angle(self.image_field)
        return self.image_field

    # Optimization method
    def opt(self):
        pass

    # Backpropagate the image plane light field to the SLM plane
    def backpropagate(self, image_field=None):
        if image_field is None:
            image_field = self.image_field

        # t = time.time()
        backprop = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(image_field), norm="ortho"))
        # print('IFFT time:' + str(time.time() - t))
        self.A = cp.abs(backprop)
        self.phi = cp.angle(backprop)

        self.slm_field = self.input * cp.exp(1j * self.phi)

    # Iterate n times and optimize at each step, keeping track of the amplitude and phase deviations at each step, as
    # well as the iteration with minimum error
    def iterate(self, n):
        for i in range(n):
            self.propagate()

            self.it.append(i + 1)
            self.psi_dev.append(self.dev_phase(waist=self.waist) / (2 * pi))
            self.B_dev.append(self.dev_amp(waist=self.waist))

            if len(self.B_dev) <= 1 or self.B_dev[-1] < cp.min(cp.array(self.B_dev[:-1])):
                self.min_dev = (i + 1, cp.copy(self.A), cp.copy(self.phi), cp.copy(self.B), cp.copy(self.psi),
                                    self.B_dev[-1], self.psi_dev[-1])

            self.opt()
            self.backpropagate()

            print('Step %d' % i)
        return self.slm_field, self.image_field

    # Plot and save the results of the IFTA run
    def save_pattern(self, name, slm, correction=False, plots=(0, 1, 2, 3, 4, 5), min=False, target=True, show=(0, 1, 2, 3, 4, 5), wavelength=413):
        if min:
            slm.ampToBMP(cp.abs(self.input).get(), name=name + '_input_amp', color=(0 in plots), show=(0 in show))
            slm.phaseToBMP(self.min_dev[2].get(), name=name + '_input_phase', correction=correction, color=(1 in plots), show=(1 in show), wavelength=wavelength)

            slm.ampToBMP(self.min_dev[3].get(), name=name + '_output_amp', color=(2 in plots), show=(2 in show))
            slm.phaseToBMP(self.min_dev[4].get(), name=name + '_output_phase', color=(3 in plots), show=(3 in show), wavelength=wavelength)
        else:
            slm.ampToBMP(cp.abs(self.slm_field).get(), name=name + '_input_amp', color=(0 in plots), show=(0 in show))
            slm.phaseToBMP(cp.angle(self.slm_field).get(), name=name + '_input_phase', correction=correction, color=(1 in plots), show=(1 in show), wavelength=wavelength)

            slm.ampToBMP(cp.abs(self.image_field).get(), name=name + '_output_amp', color=(2 in plots), show=(2 in show))
            slm.phaseToBMP(cp.angle(self.image_field).get(), name=name + '_output_phase', color=(3 in plots), show=(3 in show), wavelength=wavelength)

        if target:
            slm.ampToBMP(cp.abs(self.target).get(), name=name + '_target_amp', color=(4 in plots), show=(4 in show))
            slm.phaseToBMP(cp.angle(self.target).get(), name=name + '_target_phase', color=(5 in plots), show=(5 in show), wavelength=wavelength)

    # Take the average of the field in a circle of radius centered at pos
    def avg(self, field, pos, radius):
        avg = 0
        pts = 0
        for x in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
            for y in range(int(-radius * self.size[0]), int(radius * self.size[0]) + 1):
                if ((x / self.size[0]) ** 2 + (y / self.size[0]) ** 2) <= radius ** 2:
                    avg += field[x + pos[0]][y + pos[1]]
                    pts += 1
        return avg / pts

    # Return the phase and amplitude of set of beams at positions spots
    def beams(self, field=None, spots=None, waist=0.01):
        if field is None:
            field = self.image_field
        if spots is None:
            spots = self.spots
        return cp.array([self.avg(field, m, waist) for m in spots])

    # Calculate phase error
    def dev_phase(self, spots=None, waist=0.01, target=None):
        if spots is None:
            spots = self.spots
        if target is None:
            target = [0 for _ in spots]
        target = cp.array(target)
        phases = cp.array([(cp.angle(self.avg(self.image_field, m, waist)) + 2 * pi) % (2 * pi) for m in spots])
        # return cp.max([cp.max([min(cp.abs(p1 - p2), 2 * pi - cp.abs(p1 - p2)) for p2 in phases]) for p1 in phases])
        # return scipy.stats.circstd(phases, high=pi, low=-pi)
        return ((cp.max(phases - target) + 2 * cp.pi) % 2 * cp.pi) / (2 * cp.pi)

    # Calculate amplitude error
    def dev_amp(self, spots=None, waist=0.01):
        if spots is None:
            spots = self.spots
        amps = cp.array([cp.abs(self.avg(cp.abs(self.image_field), m, waist)) for m in spots])
        return (cp.max(amps) - cp.min(amps)) / cp.max(amps)

    # Calculate intensity error
    def dev_intensity(self, spots=None, waist=0.01):
        if spots is None:
            spots = self.spots
        intensities = cp.array([self.avg(cp.abs(self.image_field)**2, m, waist) for m in spots])
        return (cp.max(intensities) - cp.min(intensities)) / cp.max(intensities)


# Implement the WGS algorithm (intensity control only)
class WGS(IFTA):

    def __init__(self, size=cp.array((1024, 1272)), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 wavelength=413e-9, f=100, start_phase=None, reference=None, consider_phase=False, waist=0.01):
        super().__init__(size, input, target, wavelength, f, waist)
        self.consider_phase = consider_phase

        self.input = input
        if start_phase is not None:
            self.slm_field = input * cp.exp(1j * pi * start_phase)
            self.A = cp.abs(self.slm_field)
            self.phi = cp.angle(self.slm_field)
        if reference is None:
            self.reference = cp.zeros(size)
        else:
            self.reference = reference

        self.g = cp.array([cp.ones(self.B.shape)])

        self.h = cp.array([cp.ones(self.psi.shape)])

    def opt(self):
        # print(self.spots)
        # print(self.B)
        # print(self.g.shape)
        avg_B = cp.sum(cp.array([self.B[m[0], m[1]] for m in self.spots])) / len(self.spots)
        g = cp.zeros(self.B.shape)
        for m in self.spots:
            g[m[0], m[1]] += (avg_B / self.B[m[0], m[1]]) * self.g[-1][m[0], m[1]]
        # g = cp.sum([(avg_B / self.B[-1][m[0], m[1]])[m[0], m[1]] for m in self.spots]) * self.g[-1]
        self.g = cp.append(self.g, [g], axis=0)


        if self.consider_phase:
            avg_psi = cp.sum(cp.array([self.psi[m[0], m[1]] for m in self.spots])) / len(self.spots)
            h = cp.zeros(self.psi.shape)
            for m in self.spots:
                h[m[0], m[1]] += (avg_psi / self.psi[m[0], m[1]]) * self.h[-1][m[0], m[1]]
            self.h = cp.append(self.h, [h], axis=0)

    def backpropagate(self, image_field=None):
        # t = time.time()
        if not self.consider_phase:
            backprop = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(self.g[-1] * self.target * cp.exp(1j * self.psi)),
                                                   norm="ortho"))
        else:
            backprop = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(self.g[-1] * self.target * cp.exp(1j * (self.psi + self.h[-1]))),
                                                   norm="ortho"))
        # print('IFFT time:' + str(time.time() - t))
        self.A = cp.abs(backprop)
        self.phi = cp.angle(backprop)

        self.slm_field = self.input * cp.exp(1j * self.phi)

    def iterate(self, n):
        for i in range(n):
            self.propagate()

            self.it.append(i + 1)
            self.psi_dev.append(self.dev_phase(waist=self.waist) / pi)
            self.B_dev.append(self.dev_amp(waist=self.waist))

            if len(self.B_dev) <= 1 or self.B_dev[-1] < cp.min(cp.array(self.B_dev[:-1])):
                self.min_dev = (i + 1, cp.copy(self.A), cp.copy(self.phi), cp.copy(self.B), cp.copy(self.psi),
                                    self.B_dev[-1], self.psi_dev[-1])

            self.opt()
            self.backpropagate()

            print('Step %d' % i)
        return self.slm_field, self.image_field


# Unfinished IFTA
class CostOptimizer(WGS):

    def __init__(self, size=cp.array((1024, 1272)), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 start_phase=cp.zeros((1024, 1272)), wavelength=413e-9, f=100):
        super().__init__(size, input, target, wavelength, f)
        self.slm_field = input * cp.exp(1j * start_phase)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.slm_field), norm="ortho"))
        self.cost = self.cost_fxn()

    def cost_fxn(self, field=None, spots=None, waist=0.01):
        if spots is None:
            spots = self.spots
        if field is None:
            field = self.image_field

        beams = self.beams(field=field, spots=spots, waist=waist)
        avg_I = cp.mean(cp.abs(beams)**2) / cp.max(cp.abs(beams)**2)
        design_I = cp.ones(beams.shape)

        gamma = cp.sum(cp.abs(beams)**2 * design_I) / cp.sum(design_I**2)

        sigma = cp.sqrt(cp.sum(cp.abs(beams)**2 - gamma * design_I)**2 / len(beams))

        f = 0.5

        return -avg_I + f * sigma

    def opt(self):
        phi = cp.angle(self.slm_field)
        px = cp.array(cp.random.rand(2) * self.size, dtype=cp.uint)

        phi[px[0], px[1]], phi[px[0], px[1]] = cp.random.rand() * 2 * pi
        slm_field = self.input * cp.exp(1j * phi)
        image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slm_field), norm="ortho"))
        cost = self.cost_fxn(field=image_field)

        if cost < self.cost:
            self.slm_field = slm_field
            self.image_field = image_field
            self.cost = cost

        return cost

    def gradient(self, phase_1D, h=0.01):
        phase = cp.reshape(phase_1D, (-1, self.size[1]))

        current = self.cost_fxn(field=self.propagate(slm_field=phase))

        grad = cp.zeros(phase_1D.shape)

        for i in range(len(phase)):
            for j in range(len(phase[i])):
                phase[i, j] += h
                grad[i * len(phase) + j] = (self.cost_fxn(field=self.propagate(slm_field=phase)) - current) / h
                phase[i, j] -= h


# Unfinished IFTA
class OutputOutput(IFTA):
    def __init__(self, size=cp.array((1024, 1272)), input=Profile.input_gaussian(), target=Profile.gaussian_array(1, 5),
                 wavelength=413e-9, f=100, waist=0.01, beta=1):
        super().__init__(size, input, target, wavelength, f, waist)
        self.propagate()

        self.a = cp.copy(self.slm_field)
        self.b = cp.copy(self.image_field)
        self.a_prime = []
        self.b_prime = []

        self.b_driving = []

        self.beta = beta

    def propagate(self, slm_field=None):
        if slm_field is None:
            slm_field = self.slm_field

        # t = time.time()
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slm_field), norm="ortho"))
        # print('FFT time:' + str(time.time() - t))

        self.B = cp.abs(self.image_field)
        self.psi = cp.angle(self.image_field)
        return self.image_field

    def opt(self):
        self.b_driving = cp.abs(self.target) * (2 * cp.exp(1j * cp.angle(self.b_prime)) -
                                                     cp.exp(1j * cp.angle(self.b))) - self.b

        return self.b_prime + self.beta * self.b_driving

    def constraints(self, slm_field):
        if slm_field is None:
            slm_field = self.slm_field

        return cp.abs(self.input) * cp.angle(slm_field)

    def backpropagate(self, image_field=None):
        if image_field is None:
            image_field = self.image_field

        # t = time.time()
        backprop = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(image_field), norm="ortho"))
        # print('IFFT time:' + str(time.time() - t))
        self.A = cp.abs(backprop)
        self.phi = cp.angle(backprop)

        self.slm_field = self.input * cp.exp(1j * self.phi)
        return self.slm_field

    def iterate(self, n):
        for i in range(n):
            self.a = self.backpropagate(image_field=self.b)
            self.a_prime = self.constraints(slm_field=self.a)
            self.b_prime = self.propagate(slm_field=self.a_prime)
            self.b = self.opt()

            self.slm_field = self.a_prime
            self.A = cp.abs(self.slm_field)
            self.phi = cp.angle(self.slm_field)

            self.image_field = self.b_prime
            self.B = cp.abs(self.image_field)
            self.psi = cp.angle(self.image_field)

            self.it.append(i + 1)
            self.psi_dev.append(self.dev_phase(waist=self.waist) / (2 * pi))
            self.B_dev.append(self.dev_amp(waist=self.waist))

            if len(self.B_dev) <= 1 or self.B_dev[-1] < cp.min(cp.array(self.B_dev[:-1])):
                self.min_dev = (i + 1, cp.copy(self.A), cp.copy(self.phi), cp.copy(self.B), cp.copy(self.psi),
                                self.B_dev[-1], self.psi_dev[-1])

            print('Step %d' % i)
        return self.slm_field, self.image_field


# Unfinished IFTA
class ThreeStep(IFTA):
    def __init__(self):
        super().__init__()


# Implement the algorithm in the Wu paper (phase and intensity control)
class Wu(IFTA):
    def __init__(self, size=(1024, 1272), input=Profile.input_gaussian(), target=Profile.spot_array(4, 4),
                 wavelength=413e-9, f=100, waist=0.001):
        super().__init__(size=size, input=input, target=target, wavelength=wavelength, f=f, waist=waist)

        self.p = 2 * pi * cp.random.random_sample(size)

        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

        self.S = cp.zeros(size)
        self.S[:, self.S.shape[1] // 2:] = 1

        # self.I = cp.eye(self.size[0], M=self.size[1], k=0)
        self.I = cp.ones(size)

        self.A_t = cp.abs(self.target)
        self.P_t = cp.angle(self.target)

        # Generate a mask with 0 everywhere except where the target pattern is, to divide the image into target and noise areas
        self.mask = cp.where(cp.abs(self.target) > 1e-3, cp.ones(size), cp.zeros(size))
        # slm.ampToBMP(cp.abs(self.mask), name='mask', color=True, show=False)

        # Performance trackers
        self.eff = []
        self.nonunif = []
        self.phase_err = []

        self.target_phase = [cp.angle(self.avg(field=self.target, pos=spot, radius=waist)) for spot in self.spots]
        # print(self.target_phase)

    # Execute a single iteration of the algorithm
    def step(self):
        U_c = self.image_field
        A_c = cp.abs(U_c)
        P_c = cp.angle(U_c)

        A_alpha = self.A_t * self.S + A_c * (self.I - self.S)
        P_alpha = self.P_t * self.S + P_c * (self.I - self.S)
        U_alpha = A_alpha * cp.exp(1j * P_alpha)
        u_alpha = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_alpha), norm="ortho"))
        p_alpha = cp.angle(u_alpha)

        A_beta = self.A_t * (self.I - self.S) + A_c * self.S
        P_beta = self.P_t * (self.I - self.S) + P_c * self.S
        U_beta = A_beta * cp.exp(1j * P_beta)
        u_beta = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(U_beta), norm="ortho"))
        p_beta = cp.angle(u_beta)

        self.p = cp.angle(cp.exp(1j * p_alpha) + cp.exp(1j * p_beta))
        self.slm_field = self.input * cp.exp(1j * self.p)
        self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

        self.image_field /= cp.sqrt(cp.sum(cp.abs(self.image_field)**2))

    # Execute the algorithm for N iterations
    def iterate(self, N):
        for n in range(N):
            print(str(n) + ' ', end='')
            self.step()
            self.eff.append(self.eta())
            self.nonunif.append(self.dev_amp(waist=0.001))
            self.phase_err.append(self.dev_phase(waist=0.001, target=self.target_phase))

        # self.image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(self.input * cp.exp(1j * self.p)), norm="ortho"))

    # Diffraction efficiency
    def eta(self):
        return cp.sum(cp.abs(self.mask * self.image_field)**2) / cp.sum(cp.abs(self.image_field)**2)

    # Diffraction efficiency
    def I_a(self):
        return cp.sum(self.mask * cp.abs(self.image_field)**2) / cp.sum(self.mask)

    # Calculate amplitude nonuniformity
    def nonuniformity(self):
        # I_a = self.I_a()
        # I_a = cp.sum(self.mask * cp.abs(self.image_field)**2) / cp.sum(self.mask)
        # I_t = cp.sum(self.mask * cp.abs(self.target)**2) / cp.sum(self.mask)
        # normalized_image = self.image_field * cp.sum(cp.abs(self.input)) / cp.sum(cp.abs(self.image_field))
        return cp.sum(self.mask * cp.abs(cp.abs(self.image_field) - cp.abs(self.target))**2)\
               / cp.sum(self.mask * cp.abs(self.target)**2)

    # Calculate spot nonuniformity
    def spot_nonuniformity(self):
        return

    # Calculate phase error
    def phase_error(self):
        return cp.sum(self.mask * cp.abs(self.image_field)**2 * cp.abs(cp.angle(self.image_field) - cp.angle(self.target)))\
               / cp.sum(self.mask * cp.abs(self.image_field)**2 * cp.pi)

# def dev_phase(image_field, spots, waist=0.01):
#     phases = [cp.angle(self.avg(self.image_field, m, waist)) for m in spots]
#     return cp.max([cp.max([(p1 - p2) % (2 * pi) for p2 in phases]) for p1 in phases])
#     # return scipy.stats.circstd(phases, high=pi, low=-pi)
#
# def dev_amp(spots=None, waist=0.01):
#     if spots is None:
#         spots = self.spots
#     amps = [cp.abs(self.avg(self.image_field, m, waist)) for m in spots]
#     return (cp.max(amps) - cp.min(amps)) / (cp.max(amps) + cp.min(amps))


# Generate single TEM01 beam
def tem01(slm, size=(0.05, 0.05)):
    wgs = WGS(input=Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)))
    wgs.phi[-1] = slm.half()
    wgs.slm_field = wgs.input * cp.exp(1j * wgs.phi[-1])
    wgs.propagate()
    wgs.save_pattern('TEM01', slm, correction=True)

    return wgs


# WGS 2D array of beams
def array2D(slm, n=4, m=4, x_pitch=0.02, y_pitch=0.02, size=(0.05, 0.05)):
    wgs = WGS(input=Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)),
              target=Profile.spot_array(m, n, x_pitch=y_pitch, y_pitch=x_pitch))
    wgs.iterate(20)
    wgs.save_pattern('%dx%d' % (n, m), slm)

    return wgs


# WGS 1D array of beams
def array1D(slm, it=20, tries=1, n=5, pitch=0.004, size=(0.05, 0.05), start=None, ref=None, consider_phase=False, waist=0.01, plots=(0, 1, 2, 3), add_noise=False):
    wgs = []
    min = 0
    start = [start]
    for i in range(tries):
        if add_noise:
            start.append(start[0] + (cp.random.random_sample(slm.size) - 0.5) * 0.1)
        wgs.append(WGS(input=Profile.input_gaussian(beam_size=np.array(size)),
                       target=Profile.spot_array(1, n, y_pitch=pitch, center=(0.05, 0)), start_phase=start[-1], reference=None, consider_phase=consider_phase, waist=waist))
        wgs[-1].iterate(it)
        if wgs[min].min_dev[6] > wgs[-1].min_dev[6]:
            min = i
    wgs[min].save_pattern('1x%d' % n, slm, correction=True, plots=plots, min=True)
    wgs[min].A, wgs[min].phi, wgs[min].B, wgs[min].psi = wgs[min].min_dev[1:5]
    wgs[min].slm_field = wgs[min].A * cp.exp(wgs[min].phi * 1j)
    wgs[min].image_field = wgs[min].B * cp.exp(wgs[min].psi * 1j)

    print('Minimum amplitude non-uniformity at iteration %d out of %d' % (wgs[min].min_dev[0], it))
    print('Min Amplitude non-uniformity: %f' % wgs[min].min_dev[5])
    print('Phase non-uniformity: %f*Pi' % wgs[min].min_dev[6])
    print('Phases in last iteration: ' + str([cp.angle(wgs[min].avg(wgs[min].image_field, m, waist)) / pi for m in wgs[min].spots]) + ' * pi')

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


# WGS 2D array of TEM01 beams
def tem01_2D(slm, n=1, m=5, wgs=None, size=(0.05, 0.05)):
    if wgs is None:
        wgs = array2D(slm, n, m)

    wgs2 = WGS(input=Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)))
    wgs2.phi[-1] = slm.add(wgs.phi[-1], slm.half())
    wgs2.slm_field = wgs2.input * cp.exp(1j * wgs2.phi[-1])
    wgs2.propagate()
    wgs2.save_pattern('%dx%dTEM01' % (n, m), slm)
    slm.phaseToBMP(wgs2.phi[-1], name='%dx%dTEM01_input_phase' % (n, m), correction=True)


# Simulate interference pattern of WGS beam array
def interfere_wgs(slm):
    array1x5 = array1D(slm, 5, pitch=0.1)
    reference = Profile(field=Profile.input_gaussian(beam_type=0))
    interference = Profile(field=reference.field + array1x5.image_field)
    reference.save(slm, 'reference')
    interference.save(slm, 'interference_wgs')
    return interference.field


# Simulate reference interference pattern
def interfere_ref(slm):
    array1x5 = Profile(field=Profile.gaussian_array(1, 5)[0]).field
    reference = Profile(field=Profile.input_gaussian(beam_type=0))
    interference = Profile(field=reference.field + array1x5)
    reference.save(slm, 'reference')
    interference.save(slm, 'interference_ref')
    return interference.field


# Generate 1D array using OutputOutput algorithm (unfinished)
def array1D_OutputOutput(slm, it=20, n=5, pitch=0.02, size=(0.05, 0.05), start=None, ref=None, waist=0.01, plots=(0, 1, 2, 3)):

    oo = OutputOutput(input=Profile.input_gaussian(beam_size=cp.array(size)),
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


# Plot the horizontal electric field gradient at the center of the image
def plot_gradient(field, coord=0, axis=0):
    # E_t = np.real(target_field[len(target_field) / 2 - 1, :].get())
    E_im = field[len(field) / 2 - 1, :].get()
    plt.figure()
    plt.plot([i for i in range(len(field[1]))], np.real(E_im), label='E field (real)')
    plt.plot([i for i in range(len(field[1]))], np.imag(E_im), label='E field (imag)')
    # plt.plot([i + 0.5 for i in range(len(target_field[1]) - 1)], [E_t[i + 1] - E_t[i] for i in range(len(E_t) - 1)],
    #          label='E gradient (target)')
    grad = [E_im[i + 1] - E_im[i] for i in range(len(E_im) - 1)]
    plt.plot([i + 0.5 for i in range(len(field[1]) - 1)], np.real(grad), label='E gradient (real)')
    plt.plot([i + 0.5 for i in range(len(field[1]) - 1)], np.imag(grad), label='E gradient (imag)')
    plt.legend()
    plt.pause(0.001)


# Generate an array of beams using the Wu algorithm
def wu(slm, N=40, M=20, n=5, plot_each=False, size=(1272, 1024), res_factor=1):

    start_time = time.time()

    # Initialize input profile
    input_profile = cp.array(Profile.input_gaussian(beam_size=(0.2, 0.2), size=np.array(size)))
    input_profile /= cp.sqrt(cp.sum(cp.abs(input_profile)**2))

    # Target array amplitudes and phases
    amps = cp.array([1. for _ in range(n)])
    phases = [0 for _ in range(n)]
    # phases = [0, pi, 0, pi, 0]

    # Performance trackers
    eff = []
    nonunif = []
    phase_err = []

    # Record each algorithm run
    wus = []

    # Run the Wu algorithm M times
    for i in range(M):
        print('Iteration: ' + str(i))

        # Generate a target array
        target = Profile.target_output_array(1, n, center=(0, 0), input_profile=input_profile.get(), x_pitch=0.016, amps=amps.get(), phases=phases, size=np.array(size))
        target[0] = cp.array(target[0])

        # print(cp.sum(cp.abs(input_profile)**2))
        # print(cp.sum(cp.abs(target[0])**2))
        # target = Profile.gaussian_array(1, 5, waist=(0.02, 0.02))
        # target[0] *= cp.exp(1j * cp.pi / 2)

        # Run the Wu algorithm
        wu = Wu(input=input_profile, target=target, size=size)
        # print(wu.target.shape)
        wu.iterate(N)
        print()
        # print(cp.sum(cp.abs(wu.image_field)**2))
        # temp = wu.image_field
        # wu.image_field = cp.abs(wu.image_field) * cp.exp(1j * cp.angle(wu.image_field) * wu.mask)
        if plot_each:
            wu.save_pattern('wu_1x5', slm, target=False)
        result = wu.beams(waist=0.001)
        amps /= cp.abs(result)**0.5
        amps /= cp.max(amps)
        print(amps)
        # phases = -1 * cp.angle(result)
        # wu.image_field = temp
        eff.append(wu.eta())
        nonunif.append(wu.dev_amp(waist=0.001))
        phase_err.append(wu.phase_error())

        wus.append(wu)

        print('Diffraction Efficiency: ' + str(eff[-1]))
        print('Amplitude nonuniformity: ' + str(nonunif[-1]))
        print('Phase error: ' + str(phase_err[-1]))
        print()

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

    # Find the minimum nonuniformity run of the Wu algorithm
    min_it = nonunif.index(min(nonunif))

    # # TEM01 modulation
    wus[min_it].slm_field = cp.abs(wus[min_it].slm_field) * cp.exp(1j * cp.array(slm.add(cp.angle(wus[min_it].slm_field).get(), slm.half())))
    wus[min_it].image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(wus[min_it].slm_field), norm="ortho"))

    print('Time to run: ' + str(time.time() - start_time))

    # Save algorithm results
    wus[min_it].save_pattern(name='wu_1x5_wide_tem01_399', slm=slm, target=False, correction=True, show=(), wavelength=399)
    # plt.ion()
    slm.ampToBMP(cp.abs(wus[min_it].mask).get(), name='wu_1x5_mask', color=True)

    # expanded_mask = cp.where(cp.abs(wus[min_it].image_field) > 1e-4, cp.ones(wus[min_it].size), cp.zeros(wus[min_it].size))
    # slm.phaseToBMP((cp.angle(wus[min_it].image_field) * expanded_mask).get(), 'wu_1x5_image_phase_mask', color=True)

    print('')
    print('----Minimum iteration----')
    print('Diffraction Efficiency: ' + str(eff[min_it]))
    print('Amplitude nonuniformity: ' + str(nonunif[min_it]))
    print('Phase error: ' + str(phase_err[min_it]))

    plot_gradient(wus[min_it].image_field)

    plt.figure()
    plt.clf()
    # print(wus[min_it].eff)
    # print(wus[min_it].nonunif)
    # print(wus[min_it].phase_err)
    # temp = cp.array(wus[min_it].eff)
    # print(temp)
    plt.plot([n for n in range(N)], cp.array(wus[min_it].eff).get(), label='Efficiency')
    plt.plot([n for n in range(N)], cp.array(wus[min_it].nonunif).get(), label='Amplitude Nonuniformity')
    plt.plot([n for n in range(N)], cp.array(wus[min_it].phase_err).get(), label='Phase error')
    plt.xlabel('Inner Iteration')
    plt.xlim(0, N)
    plt.grid(True)
    plt.legend()
    plt.pause(.001)
    plt.savefig('images/inner_convergence.png')
    # plt.draw()

    plt.figure()
    plt.clf()
    plt.plot([m for m in range(M)], cp.array(eff).get(), label='Efficiency')
    plt.plot([m for m in range(M)], cp.array(nonunif).get(), label='Amplitude Nonuniformity')
    plt.plot([m for m in range(M)], cp.array(phase_err).get(), label='Phase error')
    plt.xlabel('Outer Iteration')
    plt.xlim(0, M)
    plt.grid(True)
    plt.legend()
    plt.savefig('images/outer_convergence.png')
    plt.show()


if __name__ == '__main__':

    size = (1024, 1272)
    slm = SLM(size=size, wavelength=399)

    # array1D(slm)
    # pass

    # tem01(slm, size=(0.02, 0.02))
    input_size = (0.05, 0.05)

    # target, spots = Profile.gaussian_array(1, 5, amps=cp.exp(cp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * cp.pi * 1j), y_pitch=0.1, waist=(0.05, 0.05))
    #
    # # array4x4 = array2D(slm, 4, 4, x_pitch=0.08, y_pitch=0.08, size=(0.05, 0.05))
    # # start_phase = cp.random.random_sample(slm.size)
    # start_phase = inverse_phase(slm, color=True, target=target, spots=spots, input_size=input_size)
    # # start_phase = start_phase + cp.random.random_sample(slm.size) * 0.5
    # # start_phase = (start_phase + slm.half())
    # slm.phaseToBMP(start_phase, '1x5_start_phase', color=True)
    # array1x5 = array1D(slm, it=20, tries=1, n=5, pitch=0.1, size=input_size, consider_phase=False, plots=(2, 3), start=start_phase, add_noise=False, waist=0.05)
    #
    # phases = -1 * cp.angle(array1x5.beams())/2
    # updated_amps = cp.exp(1j * phases)
    # updated_target, spots = Profile.gaussian_array(1, 5, amps=updated_amps)
    #
    # start_phase = inverse_phase(slm, color=True, target=updated_target, spots=spots, input_size=input_size)
    # array1x5 = array1D(slm, it=20, tries=1, n=5, pitch=0.1, size=input_size, consider_phase=False, plots=(2, 3),
    #                    start=start_phase, add_noise=False, waist=0.05)




    # start_phase = slm.BMPToPhase('images/blaze_grating.bmp')
    # ifta = IFTA(input=Profile.input_gaussian(beam_size=input_size))
    # ifta.phi = start_phase
    # ifta.slm_field = cp.abs(ifta.slm_field) * cp.exp(1j * ifta.phi)
    # ifta.propagate()
    # ifta.save_pattern('blaze_grat',slm)




    # array1x5_2 = array1D(slm, it=30, n=5, pitch=0.1, size=(0.05, 0.05), start=array1x5.phi, consider_phase=True, plots=(2, 3))

    #
    # tem01_2D(slm, 1, 5, array1x5)
    # tem01_2D(slm, 4, 4, array4x4)
    # interfere_ref(slm)


    # size = (0.05, 0.05)
    # reference = Profile(Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)) +
    #                     Profile.input_gaussian(beam_type=0, beam_size=cp.array(size)))
    # interference = Profile(field=reference.field + array1x5.image_field)

    # interfere_wgs(slm)

    # array1x5_oo = array1D_OutputOutput(slm, it=100, n=5, pitch=0.1, size=(0.05, 0.05), waist=0.05, plots=(2, 3))

    wu(slm, N=40, M=20, size=size)

    # ifta = IFTA(input=Profile.input_gaussian(beam_size=(0.05, 0.05)))
    # ifta.slm_field = cp.abs(ifta.slm_field)
    # ifta.propagate(ifta.slm_field)
    # ifta.backpropagate(ifta.image_field)
    # ifta.save_pattern('test', slm)

