# Capsule networks-based pipeline for ERP-based BCI

In this code, the approach of using Capsule Networks for building ERP-based BCIs is represented.

### Capsule networks structure:
- Conv layer (16 channels, 3x3 kernel, 2x2 stride, ReLU)
- Primary capsules (16)
- Digit capsules (2)
- Flattening layer (followed by Dropout 20%)
- Fully connected layer (256, ReLU)
- Fully connected layer (1, sigmoid)

### Dataset source MOABB:
http://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014009.html#moabb.datasets.BNCI2014009

### Signal preprocessing:
- Min-max normalization
- Data augmentation

### Model training and selection:
The leave-one-subject-out (LOSO) mechanism was used: leave one subject as a test subject, and train by pooling the rest of the data. As for validation subject, the 10th subject is removed at the beginning

### Setup
To start training, download the dataset and run:
```python
python capsnet_bci.py
```

The code was implemented based on a GPU-based Tensforflow 1.14, Keras 2.2.4 (!), MNE 0.18

For any inquiries: contact @qasymjomart, or kasymmkk@gmail.com

