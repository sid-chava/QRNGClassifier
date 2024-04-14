# QRNGClassifier

The following is a collection of Python scripts and data files used for classifying and predicting what quantum device/computer input strings of quantum generated random binary raw data (QRNG) came from. Simply predicting by chance, you would get roughly. around 25% accuracy if the data is generated by 4 different QC's (which our dataset), but our model (specifically using a gradient booster classifier model with a few more data processing features such as entropy calculations and Hadamard linear spectral transform) was able to reach >78% prediciton accuracy for classifying QNRG strings to the quantum machines that generated them by mapping quantum noise and bias. 
