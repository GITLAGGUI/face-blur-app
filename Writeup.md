<b style="font-size:28px;">Face Detection and Classification Project</b>

<br><br>

<b style="font-size:20px;">Methodology</b>
<br>
The objective of this project was to design, implement, and train a Custom Convolutional Neural Network (CNN) architecture using TensorFlow/Keras and compare its performance with an established, known model (YOLOv8) on a chosen face dataset. 

For the Custom CNN, the dataset was preprocessed by resizing all images to a standardized dimension of 128x128 pixels and normalizing the pixel values between 0 and 1. To frame the problem as a regression task within the custom architecture, the target bounding box coordinates (x0, y0, x1, y1) were also normalized relative to the original image dimensions. The Custom CNN architecture consists of three convolutional blocks (each containing a Conv2D layer with ReLU activation and MaxPooling2D), followed by a flattening layer, two dense layers with dropout for regularization, and a final dense layer with 4 units and a sigmoid activation function to output the predicted bounding box coordinates. The custom model was compiled using the Adam optimizer and Mean Squared Error (MSE) as the loss function.

In parallel, the dataset was also processed using YOLOv8, a state-of-the-art, single-stage object detection model. YOLOv8 processes the entire image in one forward pass and utilizes a complex architecture featuring a CSPDarknet backbone and an anchor-free detection head to output accurate bounding boxes and confidence scores. Data augmentation techniques inherent to the ultralytics YOLOv8 pipeline were applied to improve generalization.

<br><br>

<b style="font-size:20px;">Results of Training</b>
<br>
The Custom CNN was trained on the preprocessed dataset, monitoring both the training and validation Loss (Mean Squared Error) and Mean Absolute Error (MAE). Over the epochs, the Custom CNN showed convergence, learning to identify the general location of the face within the image. However, given its simple architecture and formulation as a direct bounding box regressor, the validation loss plateaued, indicating that while it learned to predict face coordinates, it struggled with high precision.

The YOLOv8 model, fine-tuned on the same dataset, achieved exceptional results. The training curves demonstrated rapid minimization of bounding box loss and high Mean Average Precision (mAP). YOLOv8 successfully learned the robust features required to detect faces across varying scales, illuminations, and partial occlusions with high confidence.

<br><br>

<b style="font-size:20px;">Comparisons</b>
<br>
Comparing the results of our custom model against the known YOLOv8 model highlights the advancements in modern computer vision architectures. 

The Custom CNN, while successfully implemented and capable of making educated predictions regarding face placement, was limited by its lack of spatial hierarchies and feature pyramid networks. It occasionally struggled when faces were exceptionally small or when multiple faces were present in the image (since a basic regression output only handles a fixed number of predictions).

In contrast, YOLOv8 vastly outperformed the Custom CNN. YOLOv8's architecture allowed it to dynamically detect multiple faces per image with highly precise bounding boxes. The known model processed the images significantly faster during inference and demonstrated a much higher resistance to false positives. 

The final system integrates the superior YOLOv8 model into a Streamlit Web Application, allowing users to upload imagery, visually inspect the model's bounding box predictions, and seamlessly apply a Gaussian blur to the detected faces for privacy protection.
