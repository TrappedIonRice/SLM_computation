import cv2
import numpy as np
# from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
from os import system
from tkinter import filedialog
import slm
from matplotlib import pyplot as plt

# app = QApplication(sys.argv)


def select_image():
    # Open file dialog to select an image
    filename = filedialog.askopenfilename()
    return filename


def select_roi(image):
    # Display the image and prompt the user to select ROI
    fig = slm.ampToBMP(image, name='measurement', color=True, show=True)
    roi = np.array((fig.get_axes()[0].get_xlim(), np.flip(fig.get_axes()[0].get_ylim())), np.int_)
    return roi


def resize_roi(roi, target_roi):
    # Resize the ROI to match the target area
    current_area = (roi[0][1] - roi[0][0]) * (roi[1][1] - roi[1][0])
    center = np.array([(roi[0][0] + roi[0][1]) / 2, (roi[1][0] + roi[1][1]) / 2])
    target_area = (target_roi[0][1] - target_roi[0][0]) * (target_roi[1][1] - target_roi[1][0])
    scale = np.sqrt(target_area / current_area)
    new_roi = (roi - np.ones((2, 2)) * center) * scale + np.ones((2, 2)) * center
    return np.array(new_roi, np.int_)


def compute_diffraction_efficiency(slm):
    # Select single spot image
    # single_spot_image_path = select_image()
    # if not single_spot_image_path:
    #     print("Image selection canceled.")
    #     return
    #
    # # Select diffraction pattern image
    # diffraction_pattern_image_path = select_image()
    # if not diffraction_pattern_image_path:
    #     print("Image selection canceled.")
    #     return

    # Load the images
    reference = slm.BMPToAmp(select_image())
    diffraction_pattern = slm.BMPToAmp(select_image())
    diffraction_pattern -= reference
    diffraction_pattern = np.clip(diffraction_pattern, a_min=0, a_max=None)

    # Ensure the images have the same dimensions
    if reference.shape != diffraction_pattern.shape:
        raise ValueError("Images must have the same dimensions")

    # Select ROI for each image
    # single_spot_roi = select_roi(single_spot)
    diffraction_pattern_roi = select_roi(diffraction_pattern)
    # print(diffraction_pattern_roi)
    # print(diffraction_pattern.shape)
    # print(len(diffraction_pattern))

    # print(single_spot_roi)
    # print(diffraction_pattern_roi)

    # Resize the single spot ROI to match the total area of the diffraction pattern ROI
    # diff_area = diffraction_pattern_roi[2] * diffraction_pattern_roi[3]
    # single_spot_roi_resized = resize_roi(single_spot_roi, diffraction_pattern_roi)

    # Crop images based on the selected and resized ROIs
    # single_spot_cropped = single_spot[single_spot_roi[0][0]:single_spot_roi[0][1], single_spot_roi[1][0]:single_spot_roi[1][1]]

    diffraction_pattern_cropped = diffraction_pattern[diffraction_pattern_roi[1][0]:diffraction_pattern_roi[1][1], diffraction_pattern_roi[0][0]:diffraction_pattern_roi[0][1]]

    # print(np.max(single_spot_cropped))
    # print(np.max(diffraction_pattern_cropped))
    # Compute the diffraction efficiency
    # single_spot_intensity = np.sum(single_spot_cropped)
    diffraction_pattern_intensity = np.sum(diffraction_pattern_cropped)
    total_intensity = np.sum(diffraction_pattern)

    # print(single_spot_intensity)
    # print(diffraction_pattern_intensity)

    diffraction_efficiency = diffraction_pattern_intensity / total_intensity
    print(diffraction_efficiency)

    return diffraction_efficiency


# # Example usage:
# efficiency = compute_diffraction_efficiency()
#
# if efficiency is not None:
#     print(f"Diffraction Efficiency: {efficiency}")
if __name__ == '__main__':
    slm = slm.SLM()
    # intensity = slm.BMPToAmp(select_image())
    # print(select_roi(intensity))
    compute_diffraction_efficiency(slm)
