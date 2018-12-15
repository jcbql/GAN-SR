# ECE285-GAN-for-Super-Resolution

## Implement ESGAN for Super Resolution based on https://github.com/xinntao/ESRGAN
1.Dependencies: Python3 and PyTorch.
<br>
2.Change the dataset to Caltech-256.

Create a symbolic link to the Caltech256 dataset by doing the following in the linux shell
```
mkdir Caltech256
ln -s /datasets/Caltech256/* Caltech256
```

## train.ipynb used for training
1.The training set is stored in HR_data, and the validation set is stored in Val_data.
<br>
2.The ESGAN architecture is implemented in model.py and block.py, the loss function is implemented in loss.py.

## demonstration.ipynb used for demostration
1.The HR, LR and SR images are stored in HR, LR and SR respectively.
<br>
2.The trained model, pretrained model from https://github.com/xinntao/ESRGAN and fine tuned model are stored in SavedNetworks.
