
## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car-notcar.png
[image2]: ./output_images/Hog_features.png
[image3]: ./output_images/sliding_grid.png
[image4]: ./output_images/out.png
[image5]: ./output_images/initial.png
[image6]: ./output_images/heat_map.png
[image7]: ./output_images/tresholded.png
[image8]: ./output_images/labeled.png
[image9]: ./output_images/boxed.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fourth code cell of the IPython notebook, where the `get_hog_features` and `extract_features` functions introduced in classroom, have been implemented with some minor changes.

I started by reading in all the `vehicle` and `non-vehicle` images from the provided data set, without modification or data augmentation, as the dataset was quite balanced as checked in the third code cell with 8792 vehicle images and 8968 belonging to the no-vehicle category.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(15, 15)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and through a try error process checking the performance visually in the image generated in the fifth code cell and the classifier performance. Employing just the HOG features progressively adding orientations, once the addition of additional orientations started to harm the performance of the classifier (in my case this point was for 11 orientations), I started to add pixels per cell keeping the cells per block parameter untouched. Finally I started with the suggested colorspace `YUV` as any other colorspace seem to perform worse, with the exception of the `YCrCb` colorspace, with which the results were quite similar.

Finally the HOG feature extraction parameter combination that showed to be the one to obtain highest classification accuracy values, maintaining the training times acceptable, was the one shown below.


* colorspace = 'YUV'
* orient = 11
* pix_per_cell = 15
* cell_per_block = 2
* hog_channel = 'ALL'

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM, using the default parameters provided by sklearn, as can be seen in the sixth code cell under the ### CLASSIFIER TRAINING ### mark. Using just the HOG feature extraction with the parameters explained earlier a 97.778 accuracy was obtained, which has been considered and shown to be enough.

When using the color features the performance in maters of time and accuracy dramatically dropped to 81.012 when using the color histogram features, and the performance was not enough boosted compared to the extra time taken by the training process when using spatial binning of color features.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search using different layers as recommended in classroom with various overlapping calls to the modified `find_cars` function with different `ystart` ,`ystop` and `scale` parameters to cover the distant and small cars in the medium range of the image, and the larger and closer ones in the lower section. The `ystart` and `ystop` parameters where chosen setting the demo parameter of the `find_cars` function True and observing the area coverage obtained as shown in the image below, before, a table can be found with the selected parameter values for each grid.

| Grid          | ystart        | ystop    | scale |  
|:-------------:|:-------------:| :-------:|:-----:|
| 1             | 400           | 464      |1.0    |
| 2             | 420           | 484      |1.0    |
| 3             | 400           | 496      |1.5    |
| 4             | 420           | 516      |1.5    |
| 5             | 400           | 528      |2.0    |
| 6             | 430           | 558      |2.0    |
| 7             | 440           | 568      |2.5    |
| 8             | 410           | 606      |3.0    |
| 9             | 430           | 654      |3.5    |

![alt text][image3]

The method was implemented inserting the provided `sliding_window` function code into the modified `find_cars` function below the marker # Sliding window search # in the code cell 7.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 9 scales using YCrCb 3-channel HOG features discarding both spatially binned color and histograms of color in the feature vector, which provided a nice result as can be seen in the following image.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heat-map and then thresholded that map to identify vehicle positions in the code cells from 12 to 16.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heat-map.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. All the functions where extracted from the Classroom lessons.

Here's an example result showing the heat-map from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding positive detections:

![alt text][image5]

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

### Here are six frames and their corresponding heatmaps tresholded:

![alt text][image7]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image8]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image9]


As can be seen, the method is not completely robust, in the first and fifth frames clearly identifies as a car the crash-barrier. In order to increase the stability of the car detections and reduce further the number of false positives, a vehicle class was defined in the 17th cell, saving the previous 20 car detections and "balancing" the heat-map threshold of the final processing pipeline `process_video` in the cell number 18. This approach was based in the Lane class created for the previous project Advanced lane finding and internet based search.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The trained classifier and used sliding window grids seem to perform well in the friendly environment presented in the project video, where the lighting and weather conditions are excellent and the majority of the cars drive with similar speed. However, in some image regions where the grid is not as thick as in the medium area, sometimes the processing pipeline enlarges the detected car area, which could lead to unnecessary breaking to maintain the security distance. This could be solved adding new sliding window grids, but at the same time this will slow the processing pipeline to much to be implemented in real time in a future, therefore a more refined balance should be seeked between processing time and grid refinement and feature extraction parameters.

Using another approach, once detected a new car and having some detections saved on the Vehicle class, a future position could be estimated using the calculated speed of the car and apply a refined grid over the area of the estimated position. This approach could reduce the processing time while upgrading the accuracy, however when two cars
overtake could lead to tracking loss as the pipeline could miss calculate the future position of the cars.

Other probable difficulties for the implemented pipeline could be:

* Truck identification, as the trailer enclosure usually is made-up with fabric (at least in Europe), in this case the HOG-features could be confused by the absence of concentrated brightness.

* Distinction between incoming and same direction traffic, a crash situation could be miss-detected when taking close curves with incoming traffic as the car would suddenly appear in the middle of the image and getting closer, this could lead to an unnecessary breaking without a clear differentiation between lanes. In order to solve this problem, the advanced lane detection techniques could be applied discriminating rapidly if the incoming cars are in the supposed path or in a near crash situation.


Finally the vehicle detection implemented in this project will miss detect a large amount of cars if used in a city environment, as for example while being stoped on a red light the cars moving transversally at fast speed may not be detected accurately as happens with the incoming cars of the other side lane, where once marked the detected car position with the box, the car is not anymore in the marked position but much closer. 


```python

```
