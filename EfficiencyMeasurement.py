import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

app = QApplication(sys.argv)


def select_image():
    # Open file dialog to select an image
    file_dialog = QFileDialog()
    file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif)")
    file_dialog.setWindowTitle("Select Image File")
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    if file_dialog.exec_():
        file_path = file_dialog.selectedFiles()[0]
        return file_path
    else:
        return None


def select_roi(image, window_name="Select ROI"):
    # Display the image and prompt the user to select ROI
    cv2.imshow(window_name, image)
    roi = cv2.selectROI(window_name, image, fromCenter=False)
    cv2.destroyWindow(window_name)
    return roi


def resize_roi(roi, target_area):
    # Resize the ROI to match the target area
    current_area = roi[2] * roi[3]
    scale = np.sqrt(target_area / current_area)
    new_width = int(roi[2] * scale)
    new_height = int(roi[3] * scale)
    return (roi[0], roi[1], new_width, new_height)


def compute_diffraction_efficiency():
    # Select single spot image
    single_spot_image_path = select_image()
    if not single_spot_image_path:
        print("Image selection canceled.")
        return

    # Select diffraction pattern image
    diffraction_pattern_image_path = select_image()
    if not diffraction_pattern_image_path:
        print("Image selection canceled.")
        return

    # Load the images
    single_spot = cv2.imread(single_spot_image_path, cv2.IMREAD_GRAYSCALE)
    diffraction_pattern = cv2.imread(diffraction_pattern_image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the images have the same dimensions
    if single_spot.shape != diffraction_pattern.shape:
        raise ValueError("Images must have the same dimensions")

    # Select ROI for each image
    single_spot_roi = select_roi(single_spot, "Select ROI for Single Spot")
    diffraction_pattern_roi = select_roi(diffraction_pattern, "Select ROI for Diffraction Pattern")

    # Resize the single spot ROI to match the total area of the diffraction pattern ROI
    total_area = diffraction_pattern_roi[2] * diffraction_pattern_roi[3]
    single_spot_roi_resized = resize_roi(single_spot_roi, total_area)

    # Crop images based on the selected and resized ROIs
    single_spot_cropped = single_spot[
                          int(single_spot_roi_resized[1]):int(single_spot_roi_resized[1] + single_spot_roi_resized[3]),
                          int(single_spot_roi_resized[0]):int(single_spot_roi_resized[0] + single_spot_roi_resized[2])]

    diffraction_pattern_cropped = diffraction_pattern[int(diffraction_pattern_roi[1]):int(
        diffraction_pattern_roi[1] + diffraction_pattern_roi[3]),
                                  int(diffraction_pattern_roi[0]):int(
                                      diffraction_pattern_roi[0] + diffraction_pattern_roi[2])]

    # Compute the diffraction efficiency
    single_spot_intensity = np.sum(single_spot_cropped)
    diffraction_pattern_intensity = np.sum(diffraction_pattern_cropped)

    diffraction_efficiency = diffraction_pattern_intensity / single_spot_intensity

    return diffraction_efficiency


# Example usage:
efficiency = compute_diffraction_efficiency()

if efficiency is not None:
    print(f"Diffraction Efficiency: {efficiency}")
