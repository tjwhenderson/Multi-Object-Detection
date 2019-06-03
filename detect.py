# from utils import compute_AP, NMS
# from yolo_model import yoloModel

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
# matplotlib.use('Agg')
from PIL import Image


CLASS_ID = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", \
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", \
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]



def plot_detections(imgpath, targets, predictions, batch, sample, output_dir, is_demo=False):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 100)]
    bbox_colors = random.sample(colors, 40)
    
    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    img = Image.open(imgpath).convert('RGB')
    ax[0].imshow(img)
    ax[1].imshow(img)
    ax[0].set_title('Groundtruth Bounding Boxes')
    ax[1].set_title('Predicted Bounding Boxes')
    ax[0].axis('off')
    ax[1].axis('off')

    # -------------------------- #
    # Plotting groundtruth boxes #
    # -------------------------- #
    for jj in range(len(targets)):
        cls, x, y, w, h = targets[jj,0], targets[jj,1], targets[jj,2], targets[jj,3], targets[jj,4]
        color = bbox_colors[cls]
        # Create a Rectangle patches
        bbox = patches.Rectangle((x, y), w, h, linewidth=2, \
                                 edgecolor=color, facecolor='none')
        # Add the bbox to the plot
        ax[0].add_patch(bbox)
        # Add label
        ax[0].text(x, y, s=CLASS_ID[cls], color='white', \
                 verticalalignment='top',bbox={'color': color, 'pad': 0})
        
    # ------------------------ #
    # Plotting predicted boxes #
    # ------------------------ #
    for jj in range(len(predictions)):
        cls, x, y, w, h = predictions[jj,0], predictions[jj,1], \
                          predictions[jj,2], predictions[jj,3], predictions[jj,4]
        color = bbox_colors[cls+20]
        # Create a Rectangle patches
        bbox = patches.Rectangle((x, y), w, h, linewidth=2, \
                                 edgecolor=color, facecolor='none')
        # Add the bbox to the plot
        ax[1].add_patch(bbox)
        # Add label
        ax[1].text(x, y, s=CLASS_ID[cls], color='black', \
                 verticalalignment='top',bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    
    if not is_demo: # only saves the detected images if it's not for the demo
        output_file = output_dir + '/{}_{}.jpg'.format(batch, sample)
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
        plt.close()
#     else:
#         plt.close()

    
    

            

            
            


