import matplotlib.pyplot
import numpy as np
import scipy.special
import slmsuite.holography.toolbox.phase

from profile import Profile
from slm import SLM
from matplotlib import pyplot as plt
import cupy as cp
import scipy
import laserbeamsize as lbs


# Gives the complex amplitude of an arbitrary HG beam at an x, y, z position
def hermite_beam(x, y, n=0, m=1, z=0, w=0.1, R=cp.infty, zR=cp.infty, k=0, amp=1):
    return amp * scipy.special.hermite(n)(np.sqrt(2) * x / w) * cp.exp(-x**2 / w**2) * scipy.special.hermite(m)(np.sqrt(2) * y / w) * cp.exp(-y**2 / w**2) * cp.exp(-1j * (k * z - (1 + n + m) * cp.arctan(z / zR) + k * (x**2 + y**2) / (2 * R)))


# Fill a plane with an n, m mode HG beam
def temnm(slm, n=0, m=6, amp=1, w=(0.03, 0.1)):
    field = cp.fromfunction(lambda i, j: amp * scipy.special.hermite(n)(np.sqrt(2) * (2 * (i / slm.size[0] - 0.5)) / w[0]) * np.exp(-(2 * (i / slm.size[0] - 0.5))**2 / w[0]**2) * scipy.special.hermite(m)(np.sqrt(2) * (2 * (j / slm.size[1] - 0.5)) / w[1]) * np.exp(-(2 * (j / slm.size[1] - 0.5))**2 / w[1]**2), slm.size)
    return field


# propagate SLM-plane field to image plane
def propagate(slm_field):
    return cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slm_field), norm="ortho"))


# backpropagate image plane field to SLM plane
def backpropagate(image_field):
    return cp.fft.ifftshift(cp.fft.ifft2(cp.fft.ifftshift(image_field), norm="ortho"))


# Fit gaussian beam position, waist and elliptical angle
def laserbeamsizefromimage(slm, intensity):
    x, y, dx, dy, phi = lbs.beam_size(intensity)
    # print(slm.size)
    # print("The center of the beam ellipse is at (%.1f, %.1f)" % (x, y))
    # print("The ellipse diameter (closest to horizontal) is %.4f" % (dx / slm.size[1]))
    # print("The ellipse diameter (closest to   vertical) is %.4f" % (dy / slm.size[0]))
    # print("The ellipse is rotated %.0f° ccw from horizontal" % (phi * 180 / np.pi))
    # # plt.figure()
    # # lbs.plot_image_analysis(intensity)
    # plt.pause(0.001)
    return [x, y, dx / slm.size[1], dy / slm.size[0], phi]
    # lbs.beam_size_plot(intensity)
    # plt.show()


# Return the index of the first value in an array which is greater than val
def tail(a, val):
    for i in range(len(a)):
        if a[i] >= val:
            return i
    return -1


# Make an initial guess for the pupil
def initialize_pupil(slm, slm_field):

    # Find all peaks in slm_field
    peak = np.transpose(np.where(np.abs(slm_field) == np.max(np.abs(slm_field))))
    # print(np.max(np.abs(slm_field)))
    slm_field = np.transpose(slm_field)
    # print(peak)
    # print(slm.px_to_coords(peak[0]))

    # Take the reflection of the peak coordinates such that they're in the upper left quadrant
    peak = [slm.coord(-1 * np.abs(slm.px_to_coords(np.flip(pk)))) for pk in peak]
    # print(peak)
    # print(slm_field[peak[0][1], peak[0][0]])
    # print(len(slm_field))
    # print(len(slm_field))
    # print(slm.size)
    # print(slm.coord(np.array([1, 1])))

    # Take a horizontal slice of the field
    xx = slm_field[peak[0][0], :]
    # print(np.max(np.abs(xx)))
    # x0 = np.searchsorted(np.abs(xx)[:peak[0][0] + 1], np.max(np.abs(xx)) / 100, side='left')

    # Find where the distribution falls off to 1% of the maximum value
    x0 = tail(np.abs(xx), np.max(np.abs(xx)) / 100)
    # print('x0:', x0)

    # Take a vertical slice of the field
    yy = slm_field[:, peak[0][1]]
    # print(np.max(np.abs(yy)))
    # print(np.max(np.abs(yy)) / 100)
    # print(np.abs(yy)[600:640])

    # Find where the distribution falls off to 1% of the maximum value
    y0 = tail(np.abs(yy), np.max(np.abs(yy)) / 100)
    # print('y0:', y0)

    # Find the upper left corner of the domain
    r0 = np.array([y0, x0])
    # print(r0)

    # Find the lower right corner of the domain
    r1 = slm.coord(-1 * slm.px_to_coords(r0))
    # print(r1)
    r0 = np.flip(r0)
    r1 = np.flip(r1)

    # Initialize a mask to match the domain
    mask = cp.zeros(slm.size)
    # mask = np.random.random(slm.size)
    mask[r0[0]:r1[0], r0[1]:r1[1]] = 1
    # print(np.array([r0, r1]))
    # print(r0)
    # print(np.flip(peak[0]))
    return [r0, r1], np.flip(peak[0])


# Calculate rms error in a given domain of interest for evaluating performance
def rms(field, target, domain, gamma=1):
    # area = (domain[1][0] - domain[0][0]) * (domain[1][1] - domain[0][1])
    # print(np.sum(field))
    # print(np.sum(np.abs(target)))

    # Crop field and target to the domain of interest
    field = field[domain[0][0]:domain[1][0], domain[0][1]:domain[1][1]]
    target = target[domain[0][0]:domain[1][0], domain[0][1]:domain[1][1]]
    # area = len(field) * len(field[0])
    # print(np.sum(np.abs(field - gamma * target)**2))

    # Calculate the rms error in this domain
    return np.sqrt(np.average(np.abs(field - gamma * target)**2))


# Calculate rms error for a given pupil
def evaluate_pupil(input_field, target_image_field, domain, mask):
    inverse_mask = cp.ones(mask.shape) - mask
    slm_field = np.abs(input_field) * np.exp(1j * (np.angle(input_field) * mask + inverse_mask * np.pi / 2))
    image_field = propagate(slm_field)
    # print(np.sum(np.abs(image_field)))
    return rms(image_field, target_image_field, domain)


# Optimize the pupil size to minimize rms error
def optimize_pupil(input_field, target_image_field, domain, peak):
    best_rms = 1
    out = np.copy(domain)
    # print(domain[0], peak)

    # Try every possible (rectangular symmetric) subdomain with upper left corner between the initial guess and peak
    for i in range(domain[0][0], peak[0]):
        for j in range(domain[0][1], peak[1]):
            r0 = np.array([i, j])
            r1 = np.flip(slm.coord(-1 * slm.px_to_coords(np.flip(r0))))
            # print(r0, r1)
            mask = cp.zeros(slm.size)
            mask[r0[0]:r1[0], r0[1]:r1[1]] = 1
            dom = np.array([r0, r1])
            err = evaluate_pupil(input_field, target_image_field, dom, mask)
            # print(err)
            if err < best_rms:
                best_rms = err
                out = [dom, mask]
                print(best_rms)
    print(domain)
    print(best_rms)
    print(out[0])
    return out


# Due to comment bloat, all commented code in the following uses one '#' while explanatory comments use '##'
if __name__ == '__main__':
    ## Initialize SLM parameters
    slm = SLM()

    ## Initialize input beam
    slm_waist = np.array([0.05, 0.05])
    input_field = cp.array(Profile.input_gaussian(beam_size=slm_waist))
    # image_waist = 1 / (np.pi * slm_waist)

    ## Find natural waist at image plane to determine target waist
    laserbeam = laserbeamsizefromimage(slm, cp.abs(propagate(input_field)).get()**2)
    image_waist = np.array([laserbeam[2], laserbeam[3]])
    # print(image_waist)

    ## Generate target field
    target_slm_field = temnm(slm, n=0, m=1, w=slm_waist)
    target_slm_field /= cp.max(target_slm_field)
    target_image_field = temnm(slm, n=0, m=1, w=image_waist)
    target_image_field /= cp.max(target_image_field)

    ## Make initial guess for pupil
    domain, peak = initialize_pupil(slm, target_slm_field.get())
    # pass
    # peak = [512, 636]

    ## Optimize the pupil
    domain, mask = optimize_pupil(input_field * cp.exp(1j * cp.angle(target_slm_field)), target_image_field, domain, peak)
    # print(pupil)

    ## Plot the optimized pupil
    slm.fieldtoBMP(mask.get(), name='mask', color=True)

    ## Generate the image field based on the optimized pupil
    inverse_mask = cp.ones(mask.shape) - mask
    slm_field = input_field * cp.exp(1j * (cp.angle(target_slm_field) * mask + inverse_mask * np.pi / 2))
    image_field = propagate(slm_field)
    # image_field /= cp.max(image_field)
    # laserbeamsizefromimage(slm, cp.abs(image_field).get()**2)
    # laserbeamsizefromimage(slm, cp.abs(target_field).get()**2)

    # field = temnm(slm, n=0, m=6)
    # field = propagate((cp.abs(cp.array(input_profile)) * cp.exp(1j * cp.angle(field))).get())


    ## Plot the input and image fields
    # slm.ampToBMP(cp.abs(input_field).get()**2, name='input_amp', color=True, show=False)
    slm.fieldtoBMP(slm_field.get(), name='input', color=True)
    # slm.ampToBMP(np.abs(slm_field.get()), name='input_amp', color=True)
    slm.phaseToBMP(np.angle(slm_field.get()), name='input_phase', color=True)

    # slm.ampToBMP(cp.abs(target_field).get()**2, name='target_amp', color=True, show=False)
    # slm.phaseToBMP(cp.angle(target_field).get(), name='target_phase', color=True, show=False)
    slm.fieldtoBMP(target_image_field.get(), name='target', color=True)

    # slm.ampToBMP(cp.abs(image_field).get()**2, name='image_amp', color=True, show=False)
    # slm.phaseToBMP(cp.angle(image_field).get(), name='image_phase', color=True, show=False)
    #
    slm.fieldtoBMP(image_field.get(), name='image', color=True)

    # image_pupil, image_mask = initialize_pupil(slm, target_image_field.get())
    # print(rms(image_field, target_image_field, image_pupil))

    # slm.phaseToBMP(cp.angle(field).get(), name='tem01_phase', color=True, show=False)

    ## Plot the target and calculated electric field gradients
    E_t = np.real(target_image_field[len(target_image_field) / 2 - 1, :].get())
    E_im = np.real(image_field[len(image_field) / 2 - 1, :].get())
    plt.figure()
    # plt.plot([i for i in range(len(field[1]))], E, label='E field')
    plt.plot([i + 0.5 for i in range(len(target_image_field[1]) - 1)], [E_t[i + 1] - E_t[i] for i in range(len(E_t) - 1)], label='E gradient (target)')
    plt.plot([i + 0.5 for i in range(len(image_field[1]) - 1)], [E_im[i + 1] - E_im[i] for i in range(len(E_im) - 1)], label='E gradient (image)')
    plt.legend()
    # plt.pause(0.001)
    plt.show()

    # hermite = scipy.special.hermite(1)
    # x = np.linspace(-3, 3, 400)
    # y = scipy.special.hermite(3)(x)
    #
    # plt.plot(x, y)
    # plt.show()

