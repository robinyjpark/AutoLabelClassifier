import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
from torchvision.models import resnet18, ResNet18_Weights

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

print('Load ResNet18 weights...')
# Import resnet weights
resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)

# Remove the classification layer
resnet18 = torch.nn.Sequential(*(list(resnet18.children())[:-1]))

# Set the model to evaluation mode
resnet18.to('cuda:2')
resnet18.eval()

# Define a function to extract features from the model
def extract_features(input_tensor):
    with torch.no_grad():
        input_tensor=input_tensor.to('cuda:2')
        features = resnet18(input_tensor)
    return features

# Preprocess the input images
def get_embeddings(loader):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert single-channel image to 3-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract features for each slice of each MRI volume
    features = []
    for input_images, labels in loader:
        input_images = input_images.to('cuda:2')
        labels = labels.to('cuda:2')
        for volume_idx in range(input_images.size(0)):
            volume_features = []
            for slice_idx in range(input_images.size(2)):
                # Preprocess the slice
                input_slice = input_images[volume_idx, :, slice_idx, :, :]
                input_slice = input_slice.unsqueeze(0)  # Add batch dimension

                # Convert tensor to PIL Image
                input_slice_pil = transforms.ToPILImage()(input_slice.squeeze())

                # Apply preprocessing
                input_slice_pil = preprocess(input_slice_pil)
                input_slice_pil = Variable(input_slice_pil)

                # Extract features
                slice_features = extract_features(input_slice_pil.unsqueeze(0))

                # Append the features to the volume_features list
                volume_features.append(slice_features.squeeze())

            # Get mean features for all slices in the volume
            volume_features = torch.mean(torch.stack(volume_features), dim=0)
            
        features.append(volume_features.flatten())
    return features

print('Extract features...')
# Extract features for the training, validation, and test sets
train_features = get_embeddings(train_loader)
val_features = get_embeddings(val_loader)
test_features = get_embeddings(test_loader)

print('Transfer tensors to CPU...')
train_features_cpu = [i.cpu() for i in train_features]
val_features_cpu = [i.cpu() for i in val_features]
test_features_cpu = [i.cpu() for i in test_features]

osclmric_encodings = {}
osclmric_encodings['train_features_cpu'] = train_features_cpu
osclmric_encodings['label_train_array'] = label_train_array

osclmric_encodings['val_features_cpu'] = val_features_cpu
osclmric_encodings['label_val_array'] = label_val_array

osclmric_encodings['test_features_cpu'] = test_features_cpu
osclmric_encodings['label_test_array'] = label_test_array

print('Save encodings...')
# Save the encodings
with open(f'{ivd_arrays_path}/osclmric_resnet_encodings.pkl', 'wb') as handle:
    pickle.dump(osclmric_encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done!')