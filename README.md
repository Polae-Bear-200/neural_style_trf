# Neural Style Transfer Project

## Project Description

This project aims to implement neural style transfer using the VGG19 model in PyTorch. The goal is to blend the content of one image with the style of another.

### Key Components

#### 1. Content and Style Images
- **Content Image**: The image whose content will be preserved.
- **Style Image**: The image whose artistic style will be transferred to the content image.
   
#### 2. VGG19 Model
I use the VGG19 model, a pre-trained convolutional neural network, to extract features from the images. The layers from which I extract features are:

- `conv1_1`
- `conv2_1`
- `conv3_1`
- `conv4_1`
- `conv5_1`

#### 3. Feature Extraction
I define a function to extract features from the specified layers of the VGG19 model. The features from these layers capture different levels of detail and texture.

#### 4. Loss Functions
##### Content Loss
The content loss measures the difference between the content of the `target` image and the `content` image, calculated as:

$ \text{Content Loss} = \frac{1}{2N} \sum_{i,j} (F_{ij}^{target} - F_{ij}^{content})^2 $

##### Style Loss
The style loss measures the difference between the style of the `target` image and the `style` image, using the Gram matrix to capture texture information. It is calculated as:

$ \text{Style Loss} = \frac{1}{4N^2M^2} \sum_{i,j} (G_{ij}^{target} - G_{ij}^{style})^2 $

#### 5. Total Loss
The total loss combines the content and style losses, balanced by weights:

$\text{Total Loss} = \alpha \cdot \text{Content Loss} + \beta \cdot \text{Style Loss} $

where $\alpha$ and $\beta $ are the weights for content and style losses, respectively.

#### 6. Optimization
Then I used the Adam optimizer to minimize the total loss by updating the `target` image.

# Required environment
##### Due to the limitations of my PC, which does not support GPU, I had to utilize Google Colab for my project. While attempting to create a website, I encountered issues running Flask in the Colab environment, resulting in errors. Consequently, I was unable to proceed with the website development as planned.

## Results

- Neural style transfer effectively combined content and style images.
- Target image optimization with VGG19 in PyTorch produced visually striking results.
- The project showcased the power of deep learning in generating unique artistic compositions.


## References

1. PyTorch documentation for image loading and transformation:
   - Torchvision Transforms: [Torchvision Transforms](https://pytorch.org/vision/stable/transforms.html)
   - Torchvision Image Loading: [Torchvision Datasets](https://pytorch.org/vision/stable/datasets.html)

2. PyTorch documentation for neural network functions:
   - PyTorch nn.Module: [PyTorch nn.Module](https://pytorch.org/docs/stable/nn.html)
   - PyTorch Pre-trained Models: [PyTorch Models](https://pytorch.org/vision/stable/models.html#torchvision.models.vgg19)

3. PyTorch documentation for optimizers:
   - PyTorch Optimizers: [PyTorch Optim](https://pytorch.org/docs/stable/optim.html)

4. PyTorch documentation for layers and feature extraction:
   - PyTorch Layers: [PyTorch Layers](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
   - PyTorch Feature Extraction: [Feature Extraction](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#feature-extraction)

5. PyTorch documentation for loss functions:
   - PyTorch Loss Functions: [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)

6. PyTorch documentation for Gram matrix calculation:
   - Gram Matrix Calculation: [Neural Transfer Using PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#the-gram-matrix)
