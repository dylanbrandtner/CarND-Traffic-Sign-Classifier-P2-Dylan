# **Traffic Sign Recognition Using Deep Learning** 

## Writeup

---
**Goal: Build a Neural Network to Recognize Traffic Signs**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/LeNet.png "LeNet"
[image3]: ./new_signs/sign1.jpg "Sign1"
[image4]: ./new_signs/sign2.jpg "Sign2"
[image5]: ./new_signs/sign3.jpg "Sign3"
[image6]: ./new_signs/sign4.jpg "Sign4"
[image7]: ./new_signs/sign5.jpg "Sign5"

---
### Files Submitted
* Ipython notebook with code: [Traffic_Sign_Classifier.ipynb](https://github.com/dylanbrandtner/CarND-Traffic-Sign-Classifier-Project-Dylan/blob/master/Traffic_Sign_Classifier.ipynb)
* HTML output of the code [Traffic_Sign_Classifier.html](https://github.com/dylanbrandtner/CarND-Traffic-Sign-Classifier-Project-Dylan/blob/master/Traffic_Sign_Classifier.html)
* A writeup report (either pdf or markdown) [README.md](https://github.com/dylanbrandtner/CarND-Traffic-Sign-Classifier-Project-Dylan/blob/master/README.md)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs data set:

| Item			         	     |  Size	    | 
|:------------------------------:|:------------:| 
| Size of training set  	     | 34799 	    | 
| Size of the validation set     | 4410  		|
| Size of test set				 | 12630		|
| Shape of a traffic sign image	 | 32, 32, 3    |
| Number of unique labels		 | 43 	   		|

#### Visualization of the dataset.

Here is a visualization of the data set. It is a bar chart showing how many samples are included for each sign class:

![alt text][image1]

I also wanted to view some of the sign data, so I added some logic in the notebook to parse the sign names CSV database, and then output 3 random samples of each sign from the training data with their associated text labels.  See [Traffic_Sign_Classifier.html](https://github.com/dylanbrandtner/CarND-Traffic-Sign-Classifier-Project-Dylan/blob/master/Traffic_Sign_Classifier.html) for this visualization.
### Design and Test a Model Architecture

#### Preprocessing 

##### Greyscaling

As a first step, I converted the images to grayscale.  When examining the data above, I noticed several minor differences in lighting/brightness which I suspect may have influenced the training operation.  Since color was not as important as the sign symbols themselves, it seemed like a good idea to remove color from the equation.     

Also, when reading the [successful study](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) of this same problem, which achieved 99.17% accuracy, they suggested grey scaling the data played a signficant part in their success.

##### Normalization

As a last step, I normalized the image data so that the data has mean zero and equal variance.  It seems this is a common image preprocessing step. 

#### Model architecture

My final model was essentially pulled from the LeNet lab. It has two convolutional layers with max pooling and RELU activation, a flattening layer, and then 3 fully connected layers with REUL activation.

Modified LeNet neural network architecture diagram from LeNet lecture: ![alt text][image2]

Below is a description of the LeNet architecture for a 32x32x1 input greyscale image with 43 output labels.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image 						| 
| Convolution 5x5  	 	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Activation function							|
| Max pooling	      	| 2x2 stride, valid padding, outputs 16x16x6 	|
| Convolution 5x5  	 	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					| Activation function							|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flattening 	      	| outputs 5x5x16=400						 	|
| Fully connected		| output 120   									|
| RELU					| Activation function							|
| Fully connected		| output 84   									|
| RELU					| Activation function							|
| Fully connected		| output 43   									|
| Softmax				| Applied to get probabilities					|
|						|												| 



#### Training

To train the model, I reused a lot of the python source code from the LeNet lab.  This code uses 20 epochs, a batch size of 128, and a learning rate of .001.  It calculates loss using softmax cross entropy.  It minimizes loss using the Adam Optimizer which is built into TensorFlow.  It then checks the model on the validation set and outputs an accuracy. 

#### Improving Validation Accuracy

My original model results only achieved a validation accuracy of about 86%, and my test accuracy was even lower.  

I suspected I was overfitting as my training accuracy was remaining high, and I recalled from the lesson that dropout was one of the best approaches for regularization.  Thus, I introduced a 50% dropout into the last 3 fully connected layers in my architecture.  I was also careful to avoid any dropout when validating the model on the validation and test sets.

Introducing 50% dropout slowed down my training rate, which makes sense as half the activations are ignored during training.  Thus, 20 epochs was no longer enough to reach my peak accuracy.  I experimented with some different amounts here and found that with 50 epochs, I leveled out at around 96% accuracy.
My final accuracy was: 
* 95.8% on the validation set
* 94.1% on the test set 


### Testing the Model on New Images

#### Five German traffic signs found on the web 

Here are five German traffic sign images that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]

I cropped them down to only include the signs.  

**Challenges for the model** 
* Image 1,2 and 5 include a lot of background noise.  
* Image 2 has some shadows.  
* Image 4 has a watermark.  
* Image 5 has another sign peeking out in the background 

#### Model's predictions

Here are the results of the prediction:

| Image			        |     Prediction	        | 
|:---------------------:|:-------------------------:| 
| Speed Limit 30km/h    | Speed Limit 30km/h 		| 
| Road Work   			| Road Work  				|
| Yield					| Yield						|
| Child Crossing	    | Child Crossing			|
| Turn Right Ahead		| Turn Right Ahead	   		|


The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.1%

#### Model certainty 

These are the top 5 softmax probabilities for each sign from the model.  

NOTE: Results are rounded to 10 decimal places.  Image 3 shows "100%" for "Yield" and 0 for the rest because the actual amounts were less than 0.0000000001%.

**Image 1: Speed limit (30km/h)**

| Prediction         	|     Probability	   | 
|:---------------------:|:--------------------:| 
| Speed limit (30km/h)  |  99.6836543083%      |
| Speed limit (20km/h)  |   0.1554675750%      |
| Speed limit (70km/h)  |   0.0655542128%      |
| Speed limit (50km/h)  |   0.0541767222%      |
| Roundabout mandatory  |   0.0112286703%      |


**Image 2: Road work**

| Prediction         			  |     Probability	     | 
|:-------------------------------:|:--------------------:| 
| Road work                       |  99.9952197075%      |
| Dangerous curve to the right    |   0.0029470650%      |
| Road narrows on the right       |   0.0005351747%      |
| End of no passing               |   0.0004788929%      |
| Ahead only                      |   0.0002219612%      |

**Image 3: Yield**

| Prediction         	|     Probability	   | 
|:---------------------:|:--------------------:| 
| Yield                 | 100.0000000000%      |
| Bumpy road            |   0.0000000000%      |
| Keep left             |   0.0000000000%      |
| Road work             |   0.0000000000%      |
| Ahead only            |   0.0000000000%      |

**Image 4: Children crossing**

| Prediction         	|     Probability	   | 
|:---------------------:|:--------------------:| 
| Children crossing     |  99.9855160713%      |
| Bicycles crossing     |   0.0081554441%      |
| Bumpy road            |   0.0038773855%      |
| Ahead only            |   0.0009513222%      |
| No entry              |   0.0005606900%      |

**Image 5: Turn right ahead**

| Prediction         	  |     Probability	     | 
|:-----------------------:|:--------------------:|
| Turn right ahead        |  99.9999403954%      |
| Stop                    |   0.0000289722%      |
| Yield                   |   0.0000089241%      |
| Ahead only              |   0.0000061633%      |
| Speed limit (60km/h)    |   0.0000054985%      |


As you can see, the model's certainty for all 5 signs was above 99% for the correct label.  If I had more time, I would have liked to see how it faired with even more challenging images. 
