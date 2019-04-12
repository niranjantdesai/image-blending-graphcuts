'''
https://stackoverflow.com/questions/6916054/how-to-crop-a-region-selected-with-mouse-click-using-python
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.widgets as widgets

img1Path = "1.png"
img2Path = "2.png"

def onSelect(eclick, erelease):
    minx, maxx, miny, maxy = float("inf"), -float("inf"), float("inf"), -float("inf")
    for x, y in zip([eclick.xdata, erelease.xdata], [eclick.ydata, erelease.ydata]):
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y
    minx, maxx, miny, maxy = int(minx), int(maxx), int(miny), int(maxy)
    ax3.imshow(arr1[miny:maxy, minx:maxx])

def onSelect2(eclick, erelease):
    minx, maxx, miny, maxy = float("inf"), -float("inf"), float("inf"), -float("inf")
    for x, y in zip([eclick.xdata, erelease.xdata], [eclick.ydata, erelease.ydata]):
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y
    minx, maxx, miny, maxy = int(minx), int(maxx), int(miny), int(maxy)
    ax4.imshow(arr2[miny:maxy, minx:maxx])

fig = plt.figure(figsize=(18, 18))

ax1 = fig.add_subplot(2, 2, 1)
im1 = Image.open(img1Path)
arr1 = np.asarray(im1)
ax1.imshow(arr1)
ax1.axis('off')
rs1 = widgets.RectangleSelector( ax1, onSelect, drawtype='box', rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True) )

ax2 = fig.add_subplot(2, 2, 2)
im2 = Image.open(img2Path)
arr2 = np.asarray(im2)
plt_image = plt.imshow(arr2)
ax2.axis('off')
rs2 = widgets.RectangleSelector( ax2, onSelect2, drawtype='box', rectprops = dict(facecolor='blue', edgecolor = 'black', alpha=0.5, fill=True) )

ax3 = fig.add_subplot(2, 2, 3)
ax3.axis('off')

ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

plt.show()






######### SELECTING IMAGES VIA USER INTERFACE ####################
# '''
# https://stackoverflow.com/questions/2261191/how-can-i-put-2-buttons-next-to-each-other
# https://www.pyimagesearch.com/2016/05/23/opencv-with-tkinter/
# '''

# from tkinter import *
# from PIL import Image
# from PIL import ImageTk
# from tkinter.filedialog import askopenfilename
# import cv2
# import numpy as np

# root = Tk()
# image1 = []
# image2 = []
# panelA = None
# panelB = None

# # def select_image_wrapper(argument):
# #     select_image(argument)

# def select_image1():
#     select_image(1)

# def select_image2():
#     select_image(2)

# def select_image(argument):
#     # grab a reference to the image panels
#     global image1, image2, panelA, panelB
 
#     # open a file chooser dialog and allow the user to select an input
#     # image
#     path = askopenfilename()
#     if argument==1:
#         try:
#             image1 = cv2.imread(path)
#             image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#         except:
#             image1 = None
#     elif argument==2:
#         try:
#             image2 = cv2.imread(path)
#             image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#         except:
#             image2 = None
    
#     if isinstance(image1, np.ndarray) and isinstance(image2, np.ndarray):
#         display_images()

# def display_images():
#     global image1, image2, panelA, panelB

#     # convert the images to PIL format...
#     image1_d = Image.fromarray(image1)
#     image2_d = Image.fromarray(image2)

#     # ...and then to ImageTk format
#     image1_d = ImageTk.PhotoImage(image1_d)
#     image2_d = ImageTk.PhotoImage(image2_d)

#     # if the panels are None, initialize them
#     if panelA is None or panelB is None:
#         # the first panel will store our original image
#         panelA = Label(image=image1_d)
#         panelA.image = image1_d
#         panelA.pack(side="left", padx=10, pady=10)
#         # panelA.grid(row=0, column=0, sticky=W)

#         # while the second panel will store the edge map
#         panelB = Label(image=image2_d)
#         panelB.image = image2_d
#         panelB.pack(side="right", padx=10, pady=10)
#         # panelB.grid(row=0, column=0, sticky=W)

#     # otherwise, update the image panels
#     else:
#         # update the pannels
#         panelA.configure(image=image1_d)
#         panelB.configure(image=image2_d)
#         panelA.image = image1_d
#         panelB.image = image2_d
 
# # create a button, then when pressed, will trigger a file chooser
# # dialog and allow the user to select an input image; then add the
# # button the GUI
# btn1 = Button(root, text="Select the first image", command=select_image1)
# btn1.pack(side="left", padx="10", pady="10")
# # btn1.grid(row=1, column=0, sticky=W)

# btn2 = Button(root, text="Select the second image", command=select_image2)
# btn2.pack(side="left", padx="10", pady="10")
# # btn2.grid(row=1, column=1, sticky=W)

# # kick off the GUI
# root.mainloop()

# # display_images()


# ########################
# # btn1.pack(side="left", fill="both", expand="yes", padx="10", pady="10")
# # btn1.pack(side="left", fill="both", expand="yes", padx="10", pady="10")