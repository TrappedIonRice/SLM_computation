import slmsuite.holography.algorithms as algorithms
import slmsuite.holography.toolbox.phase

import numpy as np
# import cupy as cp
import scipy
import matplotlib.pyplot as plt
import matplotlib

import WGS
import slm
import profile
from InversePhase import inverse_phase

matplotlib.use('QtAgg')


if __name__ == '__main__':
    slm = slm.SLM()
    # target, spots = profile.Profile.gaussian_array(1, 2, amps=np.exp(np.array([0, 0, 0, 0, 0]) * np.pi * 1j))

    # start_phase = np.random.random_sample(slm.size)
    # array1x5 = WGS.array1D(slm, it=50, tries=1, n=5, pitch=0.1, size=(0.05, 0.05), consider_phase=False, plots=(2, 3), start=start_phase)

    spot_vectors = profile.Profile.spot_array(5, 1)[1].transpose()

    size = (0.05, 0.05)

    shape = algorithms.SpotHologram.calculate_padded_shape((1272, 1024))
    # shape = (1272, 1024)
    input_prof = profile.Profile.input_gaussian(beam_size=np.array(size), size=shape)
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

    ifta = WGS.IFTA(input=input_prof, size=shape)
    ifta.image_field = farfield
    ifta.spots = np.array([[spot[1], spot[0]] for spot in spot_hologram.spot_knm.transpose()], dtype=np.uint)
    print(ifta.spots)
    print('Amplitude non-uniformity: %f' % ifta.dev_amp())
    print('Phase non-uniformity: %f*Pi rad' % (ifta.dev_phase() / (2 * np.pi)))
