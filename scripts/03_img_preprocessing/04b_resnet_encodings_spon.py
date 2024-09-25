import torch
import pickle
import random
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import Image
from torch.autograd import Variable
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms.functional import affine

ivd_arrays_path = '/work/robinpark/AutoLabelClassifier/data/osclmric_ivd_arrays/april2024_splits'

print('Load pickled arrays...')
# Load pickled arrays
with open(f'{ivd_arrays_path}/spon_arrays_dict.pkl', 'rb') as handle:
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
def extract_features(input_tensor, augment=False):
    with torch.no_grad():
        input_tensor=input_tensor.to('cuda:2')
        features = resnet18(input_tensor)
    return features

# Preprocess the input images
def get_embeddings(loader, augment=False):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert single-channel image to 3-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract features for each slice of each MRI volume
    features = []
    for input_images, labels in loader:
        _, num_slices, num_rows, num_cols = input_images[-1].shape
        max_cols = num_cols - 48
        min_cols =  48
        max_rows = num_rows - 40
        min_rows = 40

        ## AUGMENT ##
        if augment:
            # Slice shift +/- 2
            shift_slice = random.randint(-2, 2)
            input_images = np.roll(input_images, shift_slice, axis=2)
            if shift_slice > 0:
                input_images[:,:,0:shift_slice,:,:] = 0
            elif shift_slice < 0:
                input_images[:,:,num_slices+shift_slice:num_slices,:,:] = 0

            # Slice flip
            if random.random() > 0.5:
                input_images = np.flip(input_images, axis=2).copy()

            # Intensity +/- 0.1
            input_images = input_images + random.uniform(-0.1, 0.1)

            # Translation +/- 32x 24y pixels 
            shift_cols = random.randint(-32, 22)
            max_cols += shift_cols
            min_cols += shift_cols
            shift_rows = random.randint(-24, 24)
            max_rows += shift_rows
            min_rows += shift_rows
            
            # Scale +/- 0.1
            col_range = max_cols-min_cols
            row_range = max_rows-min_rows
            bb_scale = np.array(random.uniform(0.9,1.1))
            max_cols = max_cols + col_range*(bb_scale - 1.0)
            min_cols = min_cols - col_range*(bb_scale - 1.0)
            max_rows = max_rows + row_range*(bb_scale - 1.0)
            min_rows = min_rows - row_range*(bb_scale - 1.0)

            # Sanity Check
            if max_cols >= num_cols:
                min_cols = min_cols + (max_cols - num_cols)
                max_cols = num_cols - 1
            if max_rows >= num_rows:
                min_rows = min_rows + (max_rows - num_rows)
                max_rows = num_rows - 1
            if min_cols < 0:
                max_cols = max_cols + abs(min_cols)
                min_cols = 0
            if min_rows < 0:
                max_rows = max_rows + abs(min_rows)
                min_rows = 0
            max_cols = int(np.round(max_cols))
            min_cols = int(np.round(min_cols))
            max_rows = int(np.round(max_rows))
            min_rows = int(np.round(min_rows))

            # Rotation +/- 15.0 - THIS IS GIVING ERRORS
            rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), random.uniform(-15.0,15.0), 1)
            # print(np.transpose(input_images[0][0][:3], (1, 2, 0)).shape)
            input_images1 = np.transpose(cv2.warpAffine(np.transpose(input_images[0][0][:3], (1, 2, 0)), rotation_matrix, (num_cols, num_rows), flags=cv2.INTER_CUBIC))
            input_images2 = np.transpose(cv2.warpAffine(np.transpose(input_images[0][0][3:6], (1, 2, 0)), rotation_matrix, (num_cols, num_rows), flags=cv2.INTER_CUBIC))
            input_images3 = np.transpose(cv2.warpAffine(np.transpose(input_images[0][0][6:9], (1, 2, 0)), rotation_matrix, (num_cols, num_rows), flags=cv2.INTER_CUBIC))

            input_images = np.concatenate((input_images1, input_images2, input_images3), axis=0)
            # print(input_images.shape)
            input_images = cv2.resize(input_images[:, min_rows:max_rows,min_cols:max_cols], (224, 112), interpolation = cv2.INTER_CUBIC)
            # print(input_images.shape)
            
            # Transform back into original shape
            input_images = np.transpose(input_images, (2, 0, 1))[None,None,:,:,:]
            print(input_images.shape)
            input_images = torch.from_numpy(input_images).to('cuda:2')

        else: 
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
train_features = get_embeddings(train_loader, augment=False)
val_features = get_embeddings(val_loader, augment=False)
test_features = get_embeddings(test_loader, augment=False)

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
with open(f'{ivd_arrays_path}/spon_resnet_encodings.pkl', 'wb') as handle:
    pickle.dump(osclmric_encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Done!')