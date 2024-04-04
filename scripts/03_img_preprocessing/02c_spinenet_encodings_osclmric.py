import torch
import torch.nn as nn
import pickle
import numpy as np
import sys

sys.path.append('/work/robinpark/SpineNet')
import spinenet
from spinenet import SpineNet
from torch.autograd import Variable

torch.manual_seed(0)

ivd_arrays_path = '/work/robinpark/AutoLabelClassifier/data/osclmric_ivd_arrays'

print('Load pickled arrays...')
# Load pickled arrays
with open(f'{ivd_arrays_path}/osclmric_arrays_dict.pkl', 'rb') as handle:
    osclmric_array_dict = pickle.load(handle)

ivd_train_array = osclmric_array_dict['ivd_train_array']
label_train_array = osclmric_array_dict['label_train_array']

ivd_val_array = osclmric_array_dict['ivd_val_array']
label_val_array = osclmric_array_dict['label_val_array']

ivd_test_array = osclmric_array_dict['ivd_test_array']
label_test_array = osclmric_array_dict['label_test_array']

# Set up the data loaders
train_loader = torch.utils.data.DataLoader(list(zip(ivd_train_array, label_train_array)), batch_size=1, shuffle=False)
val_loader = torch.utils.data.DataLoader(list(zip(ivd_val_array, label_val_array)), batch_size=1, shuffle=False)
test_loader = torch.utils.data.DataLoader(list(zip(ivd_test_array, label_test_array)), batch_size=1, shuffle=False)

print('Load SpineNetv2 model weights...')
spnt = SpineNet(device='cuda:1', verbose=True)
spnt_model = spnt.grading_model

# Straight classifications using spinenet
def classify_ivd_v2_resnet(classificationNet, ivd, device):
    pred_ccs = torch.Tensor().to(device).long()
    pred_fsl = torch.Tensor().to(device).long()
    pred_fsr = torch.Tensor().to(device).long()
    with torch.no_grad():
        image = torch.tensor(ivd).float().to(device) # [None, None, :, :, :]
        # forward
        net_output = classificationNet(image)
        # accumulate prediction and labels
        _, predicted_ccs = net_output[2].squeeze().max(0)
        _, predicted_fsl = net_output[8].squeeze().max(0)
        _, predicted_fsr = net_output[9].squeeze().max(0)
        torch.cuda.empty_cache()

    pred_ccs = predicted_ccs.item()
    pred_fsl = predicted_fsl.item()
    pred_fsr = predicted_fsr.item()

    max = np.max([pred_ccs, pred_fsl, pred_fsr])
    return max

print('Classify using SpineNetv2...')
spnt_label_test_array = []
for i in range(len(label_test_array)):
    label = classify_ivd_v2_resnet(spnt.grading_model, ivd_test_array[i][np.newaxis,:], 'cuda:1')
    if label > 1:
        label = 1
    spnt_label_test_array.append(label)
  
spnt_label_val_array = []
for i in range(len(label_val_array)):
    label = classify_ivd_v2_resnet(spnt.grading_model, ivd_val_array[i][np.newaxis,:], 'cuda:1')
    if label > 1:
        label = 1
    spnt_label_val_array.append(label)

# List layers with fc in the name
fc_layers = [layer[0] for layer in spnt_model.named_modules() if 'fc' in layer[0]]

# Updated model to get encodings 
class ModGrading(nn.Module):
    def __init__(self):
        super(ModGrading, self).__init__()
        self.features = nn.Sequential(*list(spnt_model.children())[:-len(fc_layers)])

    def forward(self, x):
        batch_size, channels, scans, height, width = x.size() # , imgs_in_bags
        # x = x.reshape(batch_size, channels, imgs_in_bags*scans, height, width)
        x = self.features(x)
        features = x.view(batch_size, 512).squeeze()
        
        return features
    
ivd_train_encodings = []
ivd_val_encodings = []
ivd_test_encodings = []

print('Extract features...')
model = ModGrading()
model.eval()
model.to('cuda:1')
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to('cuda:1'), labels.to('cuda:1')
        outputs = model(images)
        ivd_train_encodings.append(outputs.cpu())

    for images, labels in val_loader:
        images, labels = images.to('cuda:1'), labels.to('cuda:1')
        outputs = model(images)
        ivd_val_encodings.append(outputs.cpu())

    for images, labels in test_loader:
        images, labels = images.to('cuda:1'), labels.to('cuda:1')
        outputs = model(images)
        ivd_test_encodings.append(outputs.cpu())

osclmric_encodings = {}
osclmric_encodings['ivd_train_encodings'] = ivd_train_encodings
osclmric_encodings['label_train_array'] = label_train_array

osclmric_encodings['ivd_val_encodings'] = ivd_val_encodings
osclmric_encodings['label_val_array'] = label_val_array
osclmric_encodings['spnt_label_val_array'] = spnt_label_val_array

osclmric_encodings['ivd_test_encodings'] = ivd_test_encodings
osclmric_encodings['label_test_array'] = label_test_array
osclmric_encodings['spnt_label_test_array'] = spnt_label_test_array

print('Save encodings...')
# Save the encodings
with open(f'{ivd_arrays_path}/osclmric_spinenet_encodings.pkl', 'wb') as handle:
    pickle.dump(osclmric_encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done!')