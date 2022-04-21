CDAE-C, Convolutional Denoising AutoEncoder and Classifier

Denoising Auto-Encoder for low dose chest CT

Extracting features from encoder for classification of lung nodules

This model consists of:

  FCN, Fully Convolutional Neural Network
  
  DAE, Denoing Auto-Encoder
  
  2.5D Convolutional Classifier
 
Techniques used:

1.model

  Padding for remaining image size
  
  Deconvolutional layer for upsamping
  
  Drop out and L1 regularization for avoiding overfitting
  
  Latent code concatenation for 2.5D convolution
  
2.Data

  CT projection transformation abd reconstruction for simulating Poisson noise
  
  CT resmapling for standardizing spacing of pixels
  
  Data argumentation to balance pos and neg samples
  
  Residual images for image comparison
