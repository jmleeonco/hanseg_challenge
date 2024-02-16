'''
Final (MRI creation)
'''

import SimpleITK as sitk
import numpy as np
import os
import nrrd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import nibabel as nib
import cv2

import torch
import monai.transforms as transforms

def resample(image_3d_data, mri_3d_data):
    '''
    Stage 1. resample CT image to (0.5 0.5 2.0).
    Nothing happens to mri image.

    Input
    - image_3d_data: sitk data
    - mri_3d_data: sitk data

    Output 
    - resampled_image_3d_data: sitk data
    - mri_3d_data: sitk data
    '''

    image_original_spacing = image_3d_data.GetSpacing()
    image_original_size = image_3d_data.GetSize()

    # Define the new spacing
    new_spacing = [0.5, 0.5, 2.0]

    # Resample the image (as before)
    new_size = [int(round(osz*osp/nsp)) for osz, osp, nsp in zip(image_original_size, image_original_spacing, new_spacing)]

    resampled_image_3d_data = sitk.Resample(image_3d_data, new_size, sitk.Transform(), sitk.sitkLinear,
                                    image_3d_data.GetOrigin(), new_spacing, image_3d_data.GetDirection(), 0,
                                    image_3d_data.GetPixelID())
    
    return resampled_image_3d_data, mri_3d_data

def align_mri(resampled_image_3d_data, mri_3d_data):
    '''
    Stage 2. Align MRI to resampled CT image.
    
    Output : None

    Result is saved to:
    - temp/aligned_mri.nrrd : aligned MRI
    - temp/final_metric_value.npy
    '''


    os.makedirs('my_temp', exist_ok=True)
    # base_dir = f'/drive/project/han_competition/HaN-Seg/set_1_resampled2/case_{subject:02d}/'

    # ct_path = base_dir + f'case_{subject:02d}_IMG_CT.nrrd'
    # mr_path = base_dir + f'case_{subject:02d}_IMG_MR_T1.nrrd'

    # fixed_image =  sitk.ReadImage(ct_path, sitk.sitkFloat32)
    # moving_image = sitk.ReadImage(mr_path, sitk.sitkFloat32)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=30)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(resampled_image_3d_data, 
                                                        mri_3d_data, 
                                                        sitk.Euler3DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(resampled_image_3d_data, mri_3d_data)

    moving_resampled = sitk.Resample(mri_3d_data, resampled_image_3d_data, final_transform, sitk.sitkLinear, 0.0, mri_3d_data.GetPixelID())
    sitk.WriteImage(moving_resampled, f'my_temp/aligned_mri.nrrd')
        
    final_metric_value = registration_method.GetMetricValue()
    # save metric value
    np.save(f'my_temp/final_metric_value.npy', final_metric_value)




def get_boxes(image_3d, dim=0):
    image_3d = np.clip(image_3d, -1000, 1500)
    image = np.mean(image_3d, axis=dim)
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    cv2.imwrite(f'my_temp/mean_image.jpg', image)

    # dim0
    if dim == 0:
        model = YOLO('models/dim0.pt')
    elif dim == 1:
        model = YOLO('models/dim1.pt')
    else:
        model = YOLO('models/dim2.pt')

    results = model("my_temp/mean_image.jpg")  # predict on an image
    result = results[0]
    boxes = result.boxes  # Boxes object for bounding box outputs
    # result.save(filename='result.jpg')  # save to disk
    return boxes.xyxy

def crop_image(image_3d):

    dim0_xyxy = get_boxes(image_3d,dim=0)
    dim1_xyxy = get_boxes(image_3d,dim=1)
    dim2_xyxy = get_boxes(image_3d,dim=2)



    dim0_xyxy = dim0_xyxy.cpu().numpy()[0]
    dim1_xyxy = dim1_xyxy.cpu().numpy()[0]
    dim2_xyxy = dim2_xyxy.cpu().numpy()[0]

    # Calculate start and end indices with margin for each dimension
    # Note: This simplistic approach assumes the bounding boxes directly translate to cropping indices,
    # which may need adjustment based on your specific use case and how dimensions correlate.
    x_start, x_end = min(dim0_xyxy[0], dim1_xyxy[0]), max(dim0_xyxy[2], dim1_xyxy[2])
    y_start, y_end = min(dim0_xyxy[1], dim2_xyxy[1]), max(dim0_xyxy[3], dim2_xyxy[3])
    z_start, z_end = min(dim1_xyxy[1], dim2_xyxy[0]), max(dim1_xyxy[3], dim2_xyxy[2])

    # Apply a 10% margin
    x_margin = int((x_end - x_start) * 0.1)
    y_margin = int((y_end - y_start) * 0.1)
    z_margin = int((z_end - z_start) * 0.1)

    x_start = int(max(x_start - x_margin, 0))
    x_end = int(min(x_end + x_margin, image_3d.shape[2]))
    y_start = int(max(y_start - y_margin, 0))
    y_end = int(min(y_end + y_margin, image_3d.shape[1]))
    z_start = int(max(z_start - z_margin, 0))
    z_end = int(min(z_end + z_margin, image_3d.shape[0]))

    # Crop the image and annotation using calculated indices
    data, header = nrrd.read('my_temp/aligned_mri.nrrd')
    cropped_image_3d = image_3d[z_start:z_end, y_start:y_end, x_start:x_end]
    cropped_mri_3d = data[z_start:z_end, y_start:y_end, x_start:x_end]
    
    return cropped_image_3d, cropped_mri_3d, [z_start,z_end, y_start,y_end, x_start,x_end]



def sub_inference(valid_transform, image_3d, dim):
    # cuda available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using : ', device)

    if dim == 0:
        model = torch.load('models/DICEcdice0.6943_expID679_axisx_aug6_modelUnet10_optimizerAdam_lossHayoung_lr0.0001_lrstop1e-07_lrdecay0.5_batch_size4_patience3_pretrainedNon.pth', map_location=device)
        model.eval()

        ##### x ######
        resize_transform = transforms.Compose(
            [
                transforms.EnsureChannelFirstd(keys=["image"],channel_dim=0),
                transforms.Resized(keys=["image"], spatial_size=[image_3d.shape[1], image_3d.shape[2]], mode='bilinear'),
            ]
        )
        output_3dx = np.zeros((31,) + image_3d.shape,dtype=np.float32)
        for slice_idx in (range(image_3d.shape[0])):
            image_data = image_3d[slice_idx,:,:]
            image_data = np.stack([image_data,image_data,image_data],axis=0)
            with torch.no_grad():
                transformed_data = valid_transform({"image": image_data})

                output = model(transformed_data["image"].to(device).unsqueeze(dim=0))
                output = torch.nn.Softmax(dim=1)(output)
                output = output.squeeze().detach().cpu().numpy()

                resized_output = resize_transform({"image": output})
                resized_output = resized_output['image']
            # break
            output_3dx[:,slice_idx,:,:] = resized_output
        return output_3dx
    elif dim == 1:
        model = torch.load('models/DICEcdice0.7012_expID941_axisy_aug6_modelUnet10_optimizerAdam_lossHayoung_lr0.0001_lrstop1e-07_lrdecay0.5_batch_size4_patience3_pretrainedNon.pth', map_location=device)
        model.eval()
        ##### y ######
        resize_transform = transforms.Compose(
            [
                transforms.EnsureChannelFirstd(keys=["image"],channel_dim=0),
                transforms.Resized(keys=["image"], spatial_size=[image_3d.shape[0], image_3d.shape[2]], mode='bilinear'),
            ]
        )
        output_3dy = np.zeros((31,) + image_3d.shape,dtype=np.float32)
        for slice_idx in (range(image_3d.shape[1])):
            image_data = image_3d[:,slice_idx,:]
            image_data = np.stack([image_data,image_data,image_data],axis=0)
            with torch.no_grad():
                transformed_data = valid_transform({"image": image_data})

                output = model(transformed_data["image"].to(device).unsqueeze(dim=0))
                output = torch.nn.Softmax(dim=1)(output)
                output = output.squeeze().detach().cpu().numpy()

                resized_output = resize_transform({"image": output})
                resized_output = resized_output['image']
            # break
            output_3dy[:,:,slice_idx,:] = resized_output
        return output_3dy
    elif dim ==2 :
        model = torch.load('models/DICEcdice0.7400_expID861_comb0_axisz_aug6_modelUnet10_optimizerAdam_lossHayoung_lr0.0001_lrstop1e-07_lrdecay0.5_batch_size2_patience3_pretrainedNon.pth', map_location=device)
        model.eval()

        ##### z ######
        resize_transform = transforms.Compose(
            [
                transforms.EnsureChannelFirstd(keys=["image"],channel_dim=0),
                transforms.Resized(keys=["image"], spatial_size=[image_3d.shape[0], image_3d.shape[1]], mode='bilinear'),
            ]
        )
        output_3dz = np.zeros((31,) + image_3d.shape,dtype=np.float32)
        for slice_idx in (range(image_3d.shape[2])):
            image_data = image_3d[:,:,slice_idx]
            image_data = np.stack([image_data,image_data,image_data],axis=0)
            with torch.no_grad():
                transformed_data = valid_transform({"image": image_data})

                output = model(transformed_data["image"].to(device).unsqueeze(dim=0))
                output = torch.nn.Softmax(dim=1)(output)
                output = output.squeeze().detach().cpu().numpy()

                resized_output = resize_transform({"image": output})
                resized_output = resized_output['image']
            # break
            output_3dz[:,:,:,slice_idx] = resized_output
        return output_3dz

# def sub_inference(valid_transform, image_3d, dim):
#     # cuda available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('using : ', device)

#     if dim == 0:
#         model = torch.load('models/DICEcdice0.6943_expID679_axisx_aug6_modelUnet10_optimizerAdam_lossHayoung_lr0.0001_lrstop1e-07_lrdecay0.5_batch_size4_patience3_pretrainedNon.pth', map_location=device)
#         model.eval()
#         model.to(device).half()  

#         ##### x ######
#         resize_transform = transforms.Compose(
#             [
#                 transforms.EnsureChannelFirstd(keys=["image"],channel_dim=0),
#                 transforms.Resized(keys=["image"], spatial_size=[image_3d.shape[1], image_3d.shape[2]], mode='bilinear'),
#             ]
#         )
#         output_3dx = np.zeros((31,) + image_3d.shape,dtype=np.float16)
#         for slice_idx in (range(image_3d.shape[0])):
#             image_data = image_3d[slice_idx,:,:]
#             image_data = np.stack([image_data,image_data,image_data],axis=0) #.astype(np.float16) 
#             with torch.no_grad():
#                 transformed_data = valid_transform({"image": image_data})
#                 with torch.autocast(device_type=device.type, dtype=torch.float16):
#                     output = model(transformed_data["image"].to(device).unsqueeze(dim=0).half())
#                     output = torch.nn.Softmax(dim=1)(output)
#                     output = output.squeeze().detach().cpu().numpy().astype(np.float16)

#                 resized_output = resize_transform({"image": output})
#                 resized_output = resized_output['image']
#             # break
#             output_3dx[:,slice_idx,:,:] = resized_output
#         return output_3dx
#     elif dim == 1:
#         model = torch.load('models/DICEcdice0.7012_expID941_axisy_aug6_modelUnet10_optimizerAdam_lossHayoung_lr0.0001_lrstop1e-07_lrdecay0.5_batch_size4_patience3_pretrainedNon.pth', map_location=device)
#         model.eval()
#         model.to(device).half()  

#         ##### y ######
#         resize_transform = transforms.Compose(
#             [
#                 transforms.EnsureChannelFirstd(keys=["image"],channel_dim=0),
#                 transforms.Resized(keys=["image"], spatial_size=[image_3d.shape[0], image_3d.shape[2]], mode='bilinear'),
#             ]
#         )
#         output_3dy = np.zeros((31,) + image_3d.shape,dtype=np.float16)
#         for slice_idx in (range(image_3d.shape[1])):
#             image_data = image_3d[:,slice_idx,:]
#             image_data = np.stack([image_data,image_data,image_data],axis=0) #.astype(np.float16) 
#             with torch.no_grad():
#                 transformed_data = valid_transform({"image": image_data})
#                 with torch.autocast(device_type=device.type, dtype=torch.float16):
#                     output = model(transformed_data["image"].to(device).unsqueeze(dim=0).half())
#                     output = torch.nn.Softmax(dim=1)(output)
#                     output = output.squeeze().detach().cpu().numpy().astype(np.float16)

#                 resized_output = resize_transform({"image": output})
#                 resized_output = resized_output['image']
#             # break
#             output_3dy[:,:,slice_idx,:] = resized_output
#         return output_3dy
#     elif dim ==2 :
#         model = torch.load('models/DICEcdice0.7400_expID861_comb0_axisz_aug6_modelUnet10_optimizerAdam_lossHayoung_lr0.0001_lrstop1e-07_lrdecay0.5_batch_size2_patience3_pretrainedNon.pth', map_location=device)
#         model.eval()
#         model.to(device).half()  

#         ##### z ######
#         resize_transform = transforms.Compose(
#             [
#                 transforms.EnsureChannelFirstd(keys=["image"],channel_dim=0),
#                 transforms.Resized(keys=["image"], spatial_size=[image_3d.shape[0], image_3d.shape[1]], mode='bilinear'),
#             ]
#         )
#         output_3dz = np.zeros((31,) + image_3d.shape,dtype=np.float16)
#         for slice_idx in (range(image_3d.shape[2])):
#             image_data = image_3d[:,:,slice_idx]
#             image_data = np.stack([image_data,image_data,image_data],axis=0) #.astype(np.float16) 
#             with torch.no_grad():
#                 transformed_data = valid_transform({"image": image_data})
#                 with torch.autocast(device_type=device.type, dtype=torch.float16):
#                     output = model(transformed_data["image"].to(device).unsqueeze(dim=0).half())
#                     output = torch.nn.Softmax(dim=1)(output)
#                     output = output.squeeze().detach().cpu().numpy().astype(np.float16)

#                 resized_output = resize_transform({"image": output})
#                 resized_output = resized_output['image']
#             # break
#             output_3dz[:,:,:,slice_idx] = resized_output
#         return output_3dz
    
def inference(image_3d):

    ##### if 3 channels
    valid_transform = transforms.Compose(
        [
            transforms.EnsureChannelFirstd(keys=["image"],channel_dim=0),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=1500,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            transforms.Resized(keys=["image"], spatial_size=[512, 512], mode='bilinear'),
        ]
    )
    output_3d = np.zeros((31,) + image_3d.shape,dtype=np.float32)
    output_3d += sub_inference(valid_transform, image_3d, 0)
    output_3d += sub_inference(valid_transform, image_3d, 1)
    output_3d += sub_inference(valid_transform, image_3d, 2)

    output_3d = np.argmax(output_3d,axis=0)
    return output_3d




def resample_output(output_3d, image_3d_data, resampled_image_3d,resampled_image_3d_data, coordinates):
    output_full_3d = np.zeros(resampled_image_3d.shape)
    output_full_3d[coordinates[0]:coordinates[1], coordinates[2]:coordinates[3], coordinates[4]:coordinates[5]] = output_3d

    # save as nrrd
    output_full_3d = np.transpose(output_full_3d, (2, 1, 0))
    output_nrrd = sitk.GetImageFromArray(output_full_3d)
    output_nrrd.CopyInformation(resampled_image_3d_data)

    # resample to original size
    output_original_spacing = output_nrrd.GetSpacing()
    output_original_size = output_nrrd.GetSize()

    # Define the new spacing
    new_spacing = image_3d_data.GetSpacing()

    # Resample the image (as before)
    new_size = [int(round(osz*osp/nsp)) for osz, osp, nsp in zip(output_original_size, output_original_spacing, new_spacing)]
    

    resampled_output_3d_data = sitk.Resample(output_nrrd, new_size, sitk.Transform(), sitk.sitkNearestNeighbor,
                                            output_nrrd.GetOrigin(), new_spacing, output_nrrd.GetDirection(), 0,
                                            output_nrrd.GetPixelID())
    return resampled_output_3d_data
