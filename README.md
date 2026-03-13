# Intel Image Classification 
## Custom Convolutional Neural Network (CNN) & Edge Deployment Preparation
### By Muhammad Auffa Hakim Aditya

This project presents an end-to-end Deep Learning pipeline built with TensorFlow and Keras to classify natural scenes from the Intel Image Classification dataset. Going beyond basic model training, this project emphasizes production readiness by implementing custom callbacks, data augmentation, and exporting the trained model to TensorFlow Lite (TFLite) for mobile or edge deployment.

The project was developed by Muhammad Auffa Hakim Aditya to demonstrate intermediate-to-advanced Computer Vision engineering, showcasing how to build a CNN from scratch, manage learning rates dynamically, and prepare a model for environments outside of a standard Python backend.

------------------------------------------------------------

PROJECT OBJECTIVES

1. Automatically download and organize the Intel Image dataset using `kagglehub`.
2. Construct a robust data pipeline using `image_dataset_from_directory` and optimize data loading with `tf.data.AUTOTUNE`, caching, and prefetching.
3. Prevent overfitting through inline Data Augmentation (Random Flip, Rotation, Zoom) and Dropout layers.
4. Design and train a Custom CNN architecture from scratch.
5. Implement an advanced suite of Keras Callbacks:
   - EarlyStopping
   - ReduceLROnPlateau
   - ModelCheckpoint
   - A Custom Callback (`StopAtAccuracy`) to halt training once both training and validation accuracy hit 95%.
6. Evaluate model performance and visualize accuracy/loss curves.
7. Export the final model to `SavedModel` format and convert it to `TFLite` for edge deployment.
8. Perform a live inference comparison between the native Keras model and the TFLite Interpreter.

------------------------------------------------------------

DATASET INFORMATION

Source          : Kaggle (blourdhuraju/intel-image-classification-dataset)
Domain          : Computer Vision / Scenery Recognition
Input Data      : RGB Images (Resized to 150x150)
Classes         : 6 Categories (buildings, forest, glacier, mountain, sea, street)

------------------------------------------------------------

PIPELINE ARCHITECTURE

1. Data Processing:
   - Rescaling (1./255) integrated directly into the model sequence.
   - On-the-fly Data Augmentation pipeline to enrich the training data without taking up extra disk space.

2. Custom CNN Model:
   - Conv2D (32 filters) + MaxPooling2D
   - Conv2D (64 filters) + MaxPooling2D
   - Conv2D (128 filters) + MaxPooling2D
   - Conv2D (128 filters) + MaxPooling2D
   - Flatten
   - Dense (256 units, ReLU)
   - Dropout (0.5)
   - Dense (6 units, Softmax)

------------------------------------------------------------

MODEL EXPORT & DEPLOYMENT ARTIFACTS

This project prepares the model for real-world application, not just notebook execution.

Exported Files:
- best_model.keras (Saved automatically during training)
- saved_model/ (TensorFlow SavedModel directory format)
- model.tflite (Optimized model for Android/iOS/IoT devices via TensorFlow Lite)

------------------------------------------------------------

INSTALLATION

Install the required dependencies:

pip install tensorflow numpy matplotlib kagglehub

------------------------------------------------------------

HOW TO RUN

1. Clone this repository:
   git clone https://github.com/YOUR_USERNAME/intel-image-classification.git

2. Install the required libraries.
3. Run the Python script. The script will handle data downloading, model training, plot generation, and automatically create the `model.tflite` file.
4. At the end of the script, it will randomly select a test image and print the prediction confidence from both the standard model and the TFLite interpreter.

------------------------------------------------------------

AUTHOR

Muhammad Auffa Hakim Aditya

This project was developed as an exploration of:
- Computer Vision and Deep Learning
- Custom Convolutional Neural Networks (CNN)
- TensorFlow `tf.data` API Optimization
- Custom Training Callbacks
- Edge AI / TensorFlow Lite Conversion

------------------------------------------------------------

KEYWORDS 

- Muhammad Auffa Hakim Aditya
- Computer Vision
- Image Classification
- TensorFlow Keras
- Custom CNN
- TensorFlow Lite TFLite
- Deep Learning Portfolio

------------------------------------------------------------

Note:
This project utilizes an open-source dataset originally compiled by Intel for an image classification challenge.
