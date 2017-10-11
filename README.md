
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/distribution.png "Distribution"
[image2]: ./writeup/signs_1.png "Signs"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library and python methods to calculate summary statistics of the traffic signs data set, using the .shape method of numpy to obtain the sizes and image shapes, and using the np.unique() combined with len() to obtain the number of unique labels in the data set :

* The size of training set is = 34799 images
* The size of the validation set is = 4410 images
* The size of test set is = 12630 images
* The shape of a traffic sign image is = 32x32
* The number of unique classes/labels in the data set is = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the images are distributed between the 43 different image classses in the training dataset, showing up great differences between sign types. For example there are more than 2000 images belonging to the 50Km/h limit but less than 200 for the 20Km/h speed limit or general caution sign. This may be explained by the probability of finding one of those signs during a typical travel in Germany, where the probability of finding a 50km/h limit is much higher than the one of finding a general caution sign.

![alt text][image1]

Once the image distribution is characterized, one image for each class is printed as shown below to get a fast idea of the training data set images characteristics. Showing up that a great variety of brightness, contrast and blur conditions are present. As some images can not be easily distinguished the entire data set is processed in order to enhance the previously listed characteristics.

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step,  the images where sharpen because the definition of some of the images shown above was very poor, to do so the sharpen(img) function was used, defined as follows:



```python
def sharpen(img):
    gauss = cv2.GaussianBlur(img, (5,5), 20.0)   # Parameters where adjusted following documentation
    return cv2.addWeighted(img, 2, gauss, -1, 0) # Weights adjusted through try-error
```

The second step was to adjust the brightness and contrast using first a contrast adjustment and second applying a histogram equalization of each channel using the next functions


```python
def contr(img, s=1.0):                    # Parameters adjusted following documentation
    m = 130*(1-s)
    imgC = cv2.multiply(img, np.array([s]))
    return cv2.add(imgC, np.array([m]))

def Hist(img):
    imgH = img.copy() 
    imgH[:, :, 0] = cv2.equalizeHist(img[:, :, 0]) #Histogram equalization of the 3 RGB channels
    imgH[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    imgH[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return imgH
```

Once defined the processing steps a processing pipeline was set up and used to process the entire data set, as it can be notice in the following code once applied the processing pipeline to each image the resulting image was normalized in order to get a data set with zero mean and equal deviation.


```python
# Processing pipeline
def process(img):
    imgP = sharpen(img)              # Adjust blur
    imgP = contr(imgP, 0.15)    # Adjust contrast # Tuned hyperparameter trhough try-error
    return Hist(imgP)                # Apply histogram equalization

# Processing and normalization of the train data set
    
X_train_pro = []              # Processed images initialization
y_train_pro = []              # Processed data set labels initialization

for i, (image, label) in enumerate(zip(X_train, y_train)):
    zeros = np.zeros((32,32,3))
    image = process(image)        # Image processing
    norm_image = cv2.normalize(image, zeros, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # Normalization 
    X_train_pro.append(norm_image)
    y_train_pro.append(label)
    
```

[//]: # (Image References)


[image3]: ./writeup/processing.png "Processing"

Here is an example of an original image, after the sharpen process and after the entire processing with the contrast enhanced :

![alt text][image3]
 


[//]: # (Image References)

[image4]: ./writeup/Study.png "Study"

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted in the Lenet5 architecture with an added dropout layer with a final shape as follows:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 1x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 1x2 stride,  outputs 5x5x16  				    |
| Flatten               | Outputs 1x400                                 |     
| Fully connected		| Outputs 1x120      							|
| RELU  				|            									|
| Dropout				| Dropout probability 0.5						|
| Fully connected		| Outputs 1x84									|
| RELU                  |                                               |
| Fully connected       | Outputs 1x43 Logits                           |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the process described during the course, using the AdamOptimizer with a start learning rate of 0.0005, this learning rate was selected via try error starting on 0.001 and observing the performance variation based on the validation accuracy for a certain batch size and epoch number. During the training the learning rate is adjusted every 5 Epochs reducing it a 30% to achive a better accuracy.

Similarly the number of epoch used set on 30 was set searching for a balance between the training time and the performance, where before implementing the dropout layer the net started clearly overfitting the test data as after obtaining a 0.98 training accuracy the validation accuracy resulted to be 0.82. For the batch size the one sugested for the Lenet5 architecture was maintained as the variations of batch size did not appear to significantlly affect the performance.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training set accuracy of 0.994
* validation set accuracy of 0.943 
* test set accuracy of 0.927

If a well known architecture was chosen:
* What architecture was chosen? Lenet 5
* Why did you believe it would be relevant to the traffic sign application? Because the input  and purpouse where similar, having as input 32x32 RGB images with different characters to recognize.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? As long as the high validation and test accuracies prove, the Lenet5 network architecture has been a great choice, once the preprocessing of the images was implemented rising the accuracy from almost 0.890 to 0.990.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

In order to have a deeper undestanding of the net performance, 10 images where taken as screenshots from youtube videos of people driving both in country and speedways of Germany.

![alt text][image4]

The sign images where taken from theese videos

- https://www.youtube.com/watch?v=-09qULEOsVs
- https://www.youtube.com/watch?v=J9gW23wFTHI
- https://www.youtube.com/watch?v=fhFEgM_msno

Theese images represent a real life possibility of an autonomous car data aquisition in different conditions as highway, town and night shown in the videos.

The first and second  images might be difficult to classify because of the lighting conditions as they where taken from a night video. The 8th image also may be difficult as it has a shadow in th middle of the yeld sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work     			|Road work										|
| No entry					| No entry												|
| Priority road	      		| Priority road					 				|
| Road work			| Road work     							|
|No passing|No passing|
|General caution |General caution |
|Ahead only |Ahead only|
|Keep right|Keep right|
|Yeld|Yeld|
|No passing for vehicles over 3.5 metric tons|No passing for vehicles over 3.5 metric tons|
|Turn right ahead|Turn right ahead|


The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This yelds a much higher accuracy than the test set, however this can not be taken as a complete sucess, as lots of sign types are uncovered by the study images, and the ones in wich the net has problems may not be present.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

The net Clearly shows a high certainty on its decissions, as the top softmax probabilities are 1 for the selected label and 0 for the rest 4 . This may be encouraged by the use of dropout, making the net act like a consensous of diffrerent markers.




```python

```
