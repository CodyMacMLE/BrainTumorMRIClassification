# Model Card for Brain Tumor Classifier

### 1) Model Details
What is it? What architecture? Who built it? When?
</br>
### 2) Intended Use
What is this model for? What is it NOT for?
</br>
### 3) Training Data
What dataset? How many samples? What splits?
</br>
### 4) Evaluation Results
Accuracy, precision, recall, F1, confusion matrix summary (you already have these from task 1-2)
</br>
### 5) Limitations & Biases
This is the most important section for medical AI. Think about:
  - What kind of MRI scans was it trained on? Would it work on scans from a different machine?
  - What's the false negative rate? Why is that dangerous in this context?
  - Dataset size — is ~110 patients enough to generalize?
  - Class distribution — any imbalance issues?

### 6) Ethical Considerations
This is a medical classifier. What could go wrong if someone deployed it without understanding its limitations?

## Key question for you

Before you start writing: why is the false negative rate more important than the false positive rate for a brain tumor classifier? Think about this from the patient's perspective what happens in each scenario? Your answer should inform how you write the Limitations section.