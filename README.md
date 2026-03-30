# Brain Tumor Segmentation using UNET

### Problem Statement
The goal of this project is to develop a deep learning model for brain tumor segmentation using the UNET architecture. 
Brain tumor segmentation is a critical task in medical imaging, as it helps in the diagnosis, treatment planning, and 
monitoring of brain tumors. The model can be used to automatically identify and segment brain tumors from MRI scans in a
prescreening process, which can assist radiologists in making accurate diagnoses and treatment decisions.

### Dataset
The dataset used to train, validate, and test the model can be found at:
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks. The images were obtained 
from The Cancer Imaging Archive (TCIA). They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade 
glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available. 
Tumor genomic clusters and patient data is provided in data.csv file. For more information on genomic data, refer to the publication 
"Comprehensive, Integrative Genomic Analysis of Diffuse Lower-Grade Gliomas" and supplementary material available at 
https://www.nejm.org/doi/full/10.1056/NEJMoa1402121

### Model Architecture
There are two architectures being implemented in this project, the UNET architecture for segmentation and the ResNet-18 
architecture for classification. Resnet was built using transfer learning with preliminary training done on the last 
layer with the two outputs while freezing the other layers. This model was then trained again freezing all layers but 
the last two, including the last convolutional layer. The output is a binary classification of whether a tumor is present 
in the segment or not, along with a confidence score.

The UNET architecture is used for segmentation of the tumor within the brain. The output is a binary mask that highlights
the tumor area within the brain segment. The UNET architecture was built from scratch and trained on the same dataset 
as the ResNet-18 model. The output is a binary mask that highlights the tumor area within the brain segment. The UNET model
has 4 encoder layers and 4 decoder layers, with skip connections between the corresponding encoder and decoder layers. Each
encoder layer consists of two convolutional layers followed by a max pooling layer, while each decoder layer consists of 
an upsampling layer followed by two convolutional layers. The final output layer is a convolutional layer with a sigmoid 
activation function that produces the binary mask for tumor segmentation.

### Results

#### ResNet-18 Classification Model
Current performance shows that of a total 622 individual mri segments, the model predicted the proper outcome with an accuracy 
of ~84.73%. Which infers that ~15% of the time the model either gave a false positive (~6.19%) or false negative (~9.16%).
The harmful metric being the false negative in this models case at ~10%.

Rows = Actual </br>
Columns = Prediction </br>
![Confusion Matrix](outputs/images/confusion_matrix_latest.png "Confusion Matrix") </br>
![Resnet Model Metrics](outputs/images/model_metrics_latest.png "Model Metrics")

#### UNET Segmentation Model
Current performance of the UNET model shows that the model is overfitting and is not accurately segmenting the tumor area 
within the brain. The model is currently sitting at a dice score of ~0.87 on the training set and ~0.70 on the validation 
set, which indicates that the model is not generalizing well to unseen data. The loss is also significantly higher on 
the validation set (0.3127) compared to the training set (0.1415), which further indicates overfitting. The model had gone
through multiple iterations of training and hyperparameter tuning, but the results have not improved significantly. Which 
implies that the model architecture may need to be adjusted or a larger dataset may be needed to improve performance.

![Unet Model Metrics](outputs/unet-dice-loss.png "Confusion Matrix") </br>

### How to Use the Model


### Project Structure


### Limitations
**THIS MODEL IS NOT A PRODUCTION LEVEL**
This is a model built as a student project. Currently, the model is overfitting, and the heatmap is generalized over the 
whole brain segment. Neither is giving an accurate display of a tumor being present. This model version is prone to false
negatives that are harmful to misdiagnose a patient who in fact has a tumor. The goal would be to train this model to a 99.999% recall score as
missing a tumor (false negative) is more dangerous than a false alarm (false positive), therefore recall is the priority metric over precision.