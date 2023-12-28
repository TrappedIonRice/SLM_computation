import slmsuite.holography.algorithms as algorithms
import slmsuite.holography.toolbox.phase
import slmsuite.hardware.slms.slm

import numpy as np
import cupy as cp
import scipy
import matplotlib.pyplot as plt
import matplotlib

import IFTA
from slm import SLM
from profile import Profile
from InversePhase import inverse_phase

# matplotlib.use('QtAgg')


class Hamamatsu(slmsuite.hardware.slms.slm.SLM):
    """
    Template for implementing a new SLM subclass. Replace :class:`Template`
    with the desired subclass name. :class:`~slmsuite.hardware.slms.slm.SLM` is the
    superclass that sets the requirements for :class:`Template`.
    """

    def __init__(
        self,
        width,
        height,
        wav_um,
        pitch_um,
        bitdepth,
        **kwargs
    ):
        r"""
        Initialize SLM and attributes.

        Parameters
        ----------
        width : int
            Width of the SLM in pixels.
        height : int
            Height of the SLM in pixels.
        wav_um : float
            Wavelength of operation in microns.
        pitch_um : float
            Pitch of SLM pixels in microns.
        bitdepth : int
            Bits of phase resolution (e.g. 8 for 256 phase settings.)
        kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ~~~~
        These arguments, which ultimately are used to instantiate the :class:`.SLM` superclass,
        may be more accurately filled by calling the SLM's SDK functions.
        See the other implemented SLM subclasses for examples.
        """

        # Instantiate the superclass
        super().__init__(
            width,
            height,
            bitdepth=bitdepth,
            wav_um=wav_um,
            dx_um=pitch_um,
            dy_um=pitch_um,
            **kwargs
        )

        # Zero the display using the superclass `write()` function.
        self.write(None)

    def _write_hw(self, phase):
        """
        Low-level hardware interface to write ``phase`` data onto the SLM.
        When the user calls the :meth:`.SLM.write` method of
        :class:`.SLM`, ``phase`` is error checked before calling
        :meth:`_write_hw()`. See :meth:`.SLM._write_hw` for further detail.
        """
        # TODO: Insert code here to write raw phase data to the SLM.
        pass

def beam_array():
    slm = SLM()
    # target, spots = profile.Profile.gaussian_array(1, 2, amps=np.exp(np.array([0, 0, 0, 0, 0]) * np.pi * 1j))

    # start_phase = np.random.random_sample(slm.size)
    # array1x5 = WGS.array1D(slm, it=50, tries=1, n=5, pitch=0.1, size=(0.05, 0.05), consider_phase=False, plots=(2, 3), start=start_phase)

    spot_vectors = Profile.spot_array(5, 1)[1].transpose()

    size = (0.05, 0.05)

    shape = algorithms.SpotHologram.calculate_padded_shape((1272, 1024))
    # shape = (1272, 1024)
    input_prof = Profile.input_gaussian(beam_size=np.array(size), size=shape)
    print(input_prof.shape)

    spot_hologram = algorithms.SpotHologram.make_rectangular_array(shape=shape, array_shape=(5, 1), array_pitch=50,
                                                                   amp=input_prof, slm_shape=shape)

    # print(np.array([[spot[1], spot[0]] for spot in spot_hologram.spot_knm.transpose()]))

    spot_hologram.optimize(method='WGS-Kim', maxiter=50)
    nearfield = spot_hologram.extract_phase()
    farfield = spot_hologram.extract_farfield()

    slm.ampToBMP(np.abs(spot_hologram.amp), name='1x5_input_amp', color=False)
    slm.phaseToBMP(nearfield, name='1x5_input_phase', color=False, correction=False)
    slm.ampToBMP(np.abs(farfield), name='1x5_output_amp', color=False)
    slm.phaseToBMP(np.angle(farfield), name='1x5_output_phase', color=False)

    ifta = IFTA.IFTA(input=input_prof, size=shape)
    ifta.image_field = farfield
    ifta.spots = np.array([[spot[1], spot[0]] for spot in spot_hologram.spot_knm.transpose()], dtype=np.uint)
    print(ifta.spots)
    print('Amplitude non-uniformity: %f' % ifta.dev_amp())
    print('Phase non-uniformity: %f*Pi rad' % (ifta.dev_phase() / (2 * np.pi)))


if __name__ == '__main__':
    slm = SLM()
    slm_waist = np.array([0.05, 0.05])
    input_field, mesh = Profile.input_gaussian(beam_size=slm_waist, mesh=True)
    input_field = cp.array(input_field)
    input_field /= cp.max(input_field)

    slm_phase = slmsuite.holography.toolbox.phase.hermite_gaussian(mesh, 1, 1)
    slm_field = input_field * cp.exp(1j * cp.array(slm_phase))

    image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(slm_field), norm="ortho"))

    slm.fieldtoBMP(slm_field.get(), name='input', color=True)
    slm.phaseToBMP(np.angle(slm_field.get()), name='input_phase', color=True)

    slm.fieldtoBMP(image_field.get(), name='image', color=True)

    E_im = np.real(image_field[len(image_field) / 2 - 1, :].get())
    plt.figure()
    plt.plot([i + 0.5 for i in range(len(image_field[1]) - 1)], [E_im[i + 1] - E_im[i] for i in range(len(E_im) - 1)],
             label='E gradient (image)')
    plt.legend()
    # plt.pause(0.001)
    plt.show()
