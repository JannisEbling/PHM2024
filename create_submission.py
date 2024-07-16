import json
import pickle

import numpy as np
import pandas as pd
import torch

with open("predictions/y_pred_reg_2000.pkl", "rb") as f:
    reg_preds = pickle.load(f)
with open("predictions/class_predictions.pkl", "rb") as f:
    class_preds = pickle.load(f)


# Concatenate the tensors along the appropriate dimension
flattened_tensor_list = [item for sublist in reg_preds for item in sublist]
concatenated_tensor = torch.cat(flattened_tensor_list, dim=0)
# Convert the concatenated tensor to a NumPy array
numpy_array = concatenated_tensor.numpy()

print(numpy_array)
