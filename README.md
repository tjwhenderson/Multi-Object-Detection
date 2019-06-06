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
\begin{itemize}
\item demo.ipynb
  \item Runs a demo of our code and produces sample images of detected objects. 
\item Train_Model.ipynb
  \item Runs the training of our models on the training set (as described in Section 4).	
\item Evaluate_Model.ipynb	
  \item Evaluates the models by calculating the mAP on the test set.
\item make_detections.ipynb	
  \item Makes the detections of objects on images by predicting the bounding box and class.
\item Confusion_Matrix.ipynb.  
  \item Create the confusion matrix (as seen in Section 5) for the models using the test set,

\item PASCAL_Dataloader.py	
  \item Preprocesses the PASCAL dataset images and labels and splits the dataloaders into training, validation, and test.
\item YOLO_Loss.py                 
  \item Loss function used in training (as described in Section 4.1.1.)
\item yolo_model.py			
  \item YOLO v3 base model
\item yolo_model_dcnn.py.        
  \item Modified YOLO v3 model to include deformable convolutions (as described in Section 3.3)
\item bbox.py				
  \item Function to calculate the IoU (intersection over union) for overlapping bounding boxes. 
\item detect.py				
  \item Plots the bounding boxes of the detected objects for the groundtruth and predicted objects.
\item utils.py				
  \item Utility functions to include helper functions used in the project such as NMS, etc.
\item yolov3.cfg				
  \item Configuration file for the YOLO v3 model

\end{itemize}
