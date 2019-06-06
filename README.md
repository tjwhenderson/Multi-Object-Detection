# Multi-Object-Detection

Description
===========
This is a multi-object detection project that implements YOLO v3 as well as modifies it as YOLO v3.1 to include deformable convolutional layers. The project is developed by Team "We Support the Vector Machines" and composed of members Taylor Henderson, Joni DeGuzman, Shubha Bhaskaran, and Apoorva Srivastava.


Requirements
============
Download trained weights for the models at:
\hyperlink{https://drive.google.com/drive/folders/1lQaYc3YLu6g93hqbXkdLnEgGrm2Aszuk}


Code Organization
=================
demo.ipynb
  - Runs a demo of our code and produces sample images of detected objects. 
Train_Model.ipynb
  - Runs the training of our models on the training set (as described in Section 4).	
Evaluate_Model.ipynb	
  - Evaluates the models by calculating the mAP on the test set.
make_detections.ipynb	
  - Makes the detections of objects on images by predicting the bounding box and class.
Confusion_Matrix.ipynb.  
  - Create the confusion matrix (as seen in Section 5) for the models using the test set,

PASCAL_Dataloader.py	
  - Preprocesses the PASCAL dataset images and labels and splits the dataloaders into training, validation, and test.
YOLO_Loss.py                 
  - Loss function used in training (as described in Section 4.1.1.)
yolo_model.py			
  - YOLO v3 base model
yolo_model_dcnn.py.        
  - Modified YOLO v3 model to include deformable convolutions (as described in Section 3.3)
bbox.py				
  - Function to calculate the IoU (intersection over union) for overlapping bounding boxes. 
detect.py				
  - Plots the bounding boxes of the detected objects for the groundtruth and predicted objects.
utils.py				
  - Utility functions to include helper functions used in the project such as NMS, etc.
yolov3.cfg				
  - Configuration file for the YOLO v3 model

