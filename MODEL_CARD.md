## Brain Classification Model

### Model Summary
The brain classification model receives 2d mri segmentations of a patients brain and classifies whether a tumor has been 
found. The model returns a tuple with three items: the classification (label) where zero denotes no tumor and one otherwise, the 
confidence of a tumor being found, and the CAM heatmap if the model classified the image as having a tumor.

The model is built using resnet-18 with preliminary training done on the last layer with the two outputs while freezing the other layers.
A secondary train was done freezing all layers but the last two, including the last convolutional layer. Forward and backward
hooks were applied to the model during evaluation so that a heatmap of the detected tumor can be displayed overlapping
the inputted image.

The dataset used to train, validate, and test the model can be found at:
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks. The images were obtained 
from The Cancer Imaging Archive (TCIA). They correspond to 110 patients included in The Cancer Genome Atlas (TCGA) lower-grade 
glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available. 
Tumor genomic clusters and patient data is provided in data.csv file. For more information on genomic data, refer to the publication 
"Comprehensive, Integrative Genomic Analysis of Diffuse Lower-Grade Gliomas" and supplementary material available at 
https://www.nejm.org/doi/full/10.1056/NEJMoa1402121

### Intended Use
The model's intended use is as a primary screening tool for radiologists. It allows radiologists to find notable tumor areas
to start their in-depth screening. It allows radiologists to pick up key areas with more precision and speed rather than needing
to manually screen each segment.

The model is intended to receive a single top-down mri image, sliced horizontally. The model will return a prediction, confidence level,
along with a heatmap overlay on where to start their in-depth look.

### Performance
Current performance shows that of a total 622 individual mri segments, the model predicted the proper outcome with an accuracy 
of ~84.73%. Which infers that ~15% of the time the model either gave a false positive (~6.19%) or false negative (~9.16%).
The harmful metric being the false negative in this models case at ~10%.

Rows = Actual </br>
Columns = Prediction </br>
![Confusion Matrix](outputs/images/confusion_matrix_latest.png "Confusion Matrix") </br>
![Model Metrics](outputs/images/model_metrics_latest.png "Model Metrics")


### Limitations
**THIS MODEL IS NOT A PRODUCTION LEVEL**
This is a model built as a student project. Currently, the model is overfitting, and the heatmap is generalized over the 
whole brain segment. Neither is giving an accurate display of a tumor being present. This model version is prone to false
negatives that are harmful to misdiagnose a patient who in fact has a tumor. The goal would be to train this model to a 99.999% recall score as
missing a tumor (false negative) is more dangerous than a false alarm (false positive), therefore recall is the priority metric over precision.

### Ethical Considerations
Training was done on a single dataset, and one that only contained patients that have been confirmed of having a tumor somewhere
within the brain. No patients were included that had zero segments with a brain tumor. Specifications of each patient can be 
found in the .csv within the kaggle dataset.

Considering the smaller dataset demographics like age, sex, ethnicity, genetic conditions, and tumor-prone due to cancer recurrence
can not accurately be considered within the model. Additionally, the tumors within the current dataset are lower-grade, the model has not
been trained at this point on higher grade tumors and as stated above patients with no tumors at all. To increase accuracy a larger 
dataset with diverse demographics will need to be implemented into retraining the model.

Liability for the model is placed on the radiologist using the pipeline as this is a primary tool to check for tumors. Rather than
a 100% accurate result metric.

