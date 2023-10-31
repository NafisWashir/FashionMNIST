Dataset

Fashion-MNIST is a dataset of Zalando's article images-consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.
* Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.
* Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.
* The training and test data sets have 785 columns.
* The first column consists of the class labels (see above), and represents the article of clothing.
* The rest of 784 columns (1-785) contain the pixel-values of the associated image.

There are 10 different classes of images, as following:
0: T-shirt/top;
1: Trouser;
2: Pullover;
3: Dress;
4: Coat;
5: Sandal;
6: Shirt;
7: Sneaker;
8: Bag;
9: Ankle boot.
Image dimensions are 28x28.
The train set and test set are given in two separate datasets.
 Running the Code
Install the dependencies from the 'requirements.txt' file and then run the 'evaluate_model.py'.
Data Preprocessing
Data is preprocessed to a shape of (60000,28,28,1)

Model
Sequential model have been used.The Sequential model is a linear stack of layers. It can be first initialized and then we add layers using add method or we can add all layers at init stage. The layers added are as follows:
* Conv2D is a 2D Convolutional layer (i.e. spatial convolution over images). The parameters used are:
* filters - the number of filters (Kernels) used with this layer; here filters = 32;
* kernel_size - the dimmension of the Kernel: (3 x 3);
* activation - is the activation function used, in this case relu;
* kernel_initializer - the function used for initializing the kernel;
* input_shape - is the shape of the image presented to the CNN: in our case is 28 x 28 The input and output of the Conv2D is a 4D tensor.
* MaxPooling2D is a Max pooling operation for spatial data. Parameters used here are:
* pool_size, in this case (2,2), representing the factors by which to downscale in both directions;
* Conv2D with the following parameters:
* filters: 64;
* kernel_size : (3 x 3);
* activation : relu;
* MaxPooling2D with parameter:
* pool_size : (2,2);
* Conv2D with the following parameters:
* filters: 128;
* kernel_size : (3 x 3);
* activation : relu;
* Flatten. This layer Flattens the input. Does not affect the batch size. It is used without parameters;
* Dense. This layer is a regular fully-connected NN layer. It is used without parameters;
* units - this is a positive integer, with the meaning: dimensionality of the output space; in this case is: 128;
* activation - activation function : relu;
* Dense. This is the final layer (fully connected). It is used with the parameters:
* units: the number of classes (in our case 10);
* activation : softmax; for this final layer it is used softmax activation (standard for multiclass classification)
Then we compile the model, specifying as well the following parameters:
* loss;
* optimizer;
* metrics.


Model Summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 3, 3, 128)         73856     
_________________________________________________________________
dropout_2 (Dropout)          (None, 3, 3, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               147584    
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 241,546
Trainable params: 241,546
Non-trainable params: 0




Results
Model accuracy 0.9225
Test loss: 0.2025180616557598
Test accuracy: 0.9291

                         		precision    recall  f1-score   support

Class 0 (T-shirt/top)	 :       0.87      0.89      0.88      1000
    Class 1 (Trouser)	 :       0.99      0.98      0.99      1000
   Class 2 (Pullover)	 :       0.90      0.87      0.89      1000
      Class 3 (Dress) 	:       0.93      0.95      0.94      1000
       Class 4 (Coat)	 :       0.90      0.90      0.90      1000
     Class 5 (Sandal)	 :       0.99      0.98      0.99      1000
      Class 6 (Shirt) 	:       0.79      0.79      0.79      1000
    Class 7 (Sneaker) 	:       0.94      0.98      0.96      1000
        Class 8 (Bag) 	:       0.98      0.99      0.99      1000
 Class 9 (Ankle Boot) 	:       0.98      0.95      0.97      1000

            	avg / total       	        0.93      0.93      0.93     10000

