import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import cv2
import copy
import os
import argparse
from config import *

def loading_img(img_type, img_path):
    """
    :param img_type: str, "src" or "target"
    :param img_path: str, image_dir + suffix
    :return: img on cv2
    """
    img = None

    if img_type == "src":
        img = cv2.imread(os.path.join(img_path, src_filename_suffix))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif img_type == "target":
        img = cv2.imread(os.path.join(img_path, target_filename_suffix))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

class MaskCreation():
    def __init__(self):
        """ Parsing Image Path """
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', dest='image_dir', required=True, help='Saved Path of Source & Target Images.')
        args = parser.parse_args()

        self.image_dir = args.image_dir
        self.selected_image = "src"
        self.previous_coordinates = None
        self.src_image = loading_img("src", self.image_dir)
        self.src_image_copy = copy.copy(self.src_image)
        self.target_image = loading_img("target", self.image_dir)
        self.target_image_copy = copy.copy(self.target_image)
        self.mask = np.zeros(self.target_image.shape)
        self.fig = plt.figure(figsize=(15, 15))
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.ax1.set_title("Select which pixels must be included from this image")


    def onClick(self, event):
        x, y = int(event.xdata), int(event.ydata)
        new_coordinates = (x, y)

        if event.inaxes in [self.ax1]:
            if self.previous_coordinates == None:
                self.previous_coordinates = new_coordinates
                print("Setting new coordinates! {}".format(self.previous_coordinates))

            else:
                if self.selected_image == "src":
                    cv2.line(self.mask, self.previous_coordinates, new_coordinates, src_line_color, line_width)
                    cv2.line(self.src_image_copy, self.previous_coordinates, new_coordinates, src_line_color, line_width)
                    cv2.line(self.target_image_copy, self.previous_coordinates, new_coordinates, src_line_color, line_width)
                    self.ax1.imshow(self.src_image_copy)
                elif self.selected_image == "target":
                    cv2.line(self.mask, self.previous_coordinates, new_coordinates, target_line_color, line_width)
                    cv2.line(self.src_image_copy, self.previous_coordinates, new_coordinates, target_line_color, line_width)
                    cv2.line(self.target_image_copy, self.previous_coordinates, new_coordinates, target_line_color, line_width)
                    self.ax1.imshow(self.target_image_copy)
                self.previous_coordinates = new_coordinates
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def buttonClick(self, event):
        if self.selected_image == "src":
            self.selected_image = "target"
            self.ax1.imshow(self.target_image_copy)
            self.previous_coordinates = None
        elif self.selected_image == "target":
            self.selected_image = "done"
            self.mask = np.uint8(self.mask)
            self.ax1.imshow(self.mask)
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.image_dir, mask_filename_suffix), self.mask)
        elif self.selected_image == "done":
            plt.close()

    def create_mask(self):
        """ Collecting Patches from Images """
        self.ax1.imshow(self.src_image)
        self.ax1.axis('off')
        axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
        bcut = widgets.Button(axcut, 'Finish', color='red')

        self.fig.canvas.mpl_connect('button_press_event', self.onClick)
        bcut.on_clicked(self.buttonClick)

        plt.show()

if __name__ == "__main__":
    """ Calling MaskCreation Module """
    mask_creation = MaskCreation()
    mask_creation.create_mask()