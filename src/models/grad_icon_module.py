import time

import numpy as np

from typing import Any
import itertools

import torch
import torch.nn.functional as F

from src.losses.gan_loss import GANLoss
from src.losses.contextual_loss import Contextual_Loss
from src.losses.patch_nce_loss import PatchNCELoss

from src import utils
from src.models.base_module_AtoB_BtoA import BaseModule_AtoB_BtoA
from src.models.base_module_AtoB import BaseModule_AtoB
from src.models.base_module_registration import BaseModule_Registration
from src.models.components.component_grad_icon import register_pair
from monai.inferers import sliding_window_inference
import itk
import matplotlib.pyplot as plt
log = utils.get_pylogger(__name__)

class GradICONModule(BaseModule_Registration):

    def __init__(
        self,
        netR_A: torch.nn.Module,

        optimizer,
        params,
        scheduler,
        *args,
        **kwargs: Any
    ):
        super().__init__(params, *args, **kwargs)
        
        # assign generator
        self.netR_A = netR_A

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False # perform manual
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.optimizer = optimizer
        self.params = params
        self.scheduler = scheduler


    def pad_slice_to_128(self, tensor, padding_value=-1):
        # tensor shape: [batch, channel, height, width, slice]
        slices = tensor.shape[-1]
        if slices < 128:
            padding = (0, 128 - slices)  # padding only on one side
            tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        return tensor
    
    def crop_slice_to_original(self, tensor, original_slices):
        # tensor shape: [batch, channel, height, width, slice]
        return tensor[..., :original_slices]
    
    def model_step_for_train(self, batch: Any):
        moving_img, fixed_img = batch # MR, CT, syn_CT
        # print("IMAGE SHAPE MOVING", moving_img.shape)
        # print("IMAGE SHAPE FIXED", fixed_img.shape)
        # moving_img = F.interpolate(moving_img, size=(128, 128, 128), mode="trilinear",
        #                            align_corners=False)
        # fixed_img = F.interpolate(fixed_img, size=(128, 128, 128), mode="trilinear",
        #                           align_corners=False)
        original_slices = moving_img.shape[0]
        moving_img = self.pad_slice_to_128(moving_img)
        fixed_img = self.pad_slice_to_128(fixed_img)
        loss, transform_vector, warped_img = self.netR_A(moving_img, fixed_img)
        return loss

    def model_step(self, batch: Any, is_3d=False):

        moving_img, fixed_img = batch

        # print("IMAGE SHAPE MOVING", moving_img.shape)
        # print("IMAGE SHAPE FIXED", fixed_img.shape)
        if is_3d:
            moving_img_np = moving_img.cpu().detach().squeeze().numpy()
            fixed_img_np = fixed_img.cpu().detach().squeeze().numpy()

            moving_img_np = moving_img_np.transpose(2, 1, 0) # itk: D, W, H
            fixed_img_np = fixed_img_np.transpose(2, 1, 0)

            moving_img_itk = itk.image_from_array(moving_img_np)
            fixed_img_itk = itk.image_from_array(fixed_img_np)
            phi_AB, phi_BA = register_pair(self.netR_A, moving_img_itk, fixed_img_itk)

            interpolator = itk.LinearInterpolateImageFunction.New(moving_img_itk)
            warped_img = itk.resample_image_filter(moving_img_itk,
                                                    transform=phi_AB,
                                                    interpolator=interpolator,
                                                    size=itk.size(fixed_img_itk),
                                                    output_spacing=itk.spacing(fixed_img_itk),
                                                    output_direction=fixed_img_itk.GetDirection(),
                                                    output_origin=fixed_img_itk.GetOrigin()
                                                    )
            warped_img_np = itk.array_from_image(warped_img) # D, W, H -> H, W, D
            warped_img_tensor = torch.from_numpy(warped_img_np).unsqueeze(0).unsqueeze(0)
            warped_img_tensor = warped_img_tensor.to(moving_img.device)
            # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
            # axs[0].set_title("moving img")
            # axs[0].imshow(moving_img.squeeze().cpu().numpy()[..., 60], cmap="gray")
            #
            # axs[1].set_title("fixed img")
            # axs[1].imshow(fixed_img.squeeze().cpu().numpy()[..., 60], cmap="gray")
            #
            # axs[2].set_title("warped img")
            # axs[2].imshow(warped_img_tensor.squeeze().cpu().numpy()[..., 60], cmap="gray")
            # plt.show()
            return moving_img, fixed_img, warped_img_tensor

        else:
            # original_slices = evaluation_img.shape[-1] #Padding is not needed
            # moving_img_pad = self.pad_slice_to_128(moving_img)
            # fixed_img_pad = self.pad_slice_to_128(fixed_img)
            # loss, transform_vector, warped_img_pad = self.netR_A(moving_img_pad, fixed_img_pad)
            loss, transform_vector, warped_img = self.netR_A(moving_img, fixed_img)
            # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
            # axs[0].set_title("moving img")
            # axs[0].imshow(moving_img.squeeze().cpu().numpy(), cmap="gray")
            #
            # axs[1].set_title("fixed img")
            # axs[1].imshow(fixed_img.squeeze().cpu().numpy(), cmap="gray")
            #
            # axs[2].set_title("warped img")
            # axs[2].imshow(warped_img.squeeze().cpu().numpy(), cmap="gray")
            # plt.show()
            # warped_img = self.crop_slice_to_original(warped_img_pad, original_slices)
            return moving_img, fixed_img, warped_img

    # Define your model_step function to include sliding window inference

    # ###############################################################################################
    # def create_itk_transform(self, phi, ident, image_A, image_B) -> "itk.CompositeTransform":
    #
    #     # itk.DeformationFieldTransform expects a displacement field, so we subtract off the identity map.
    #     disp = (phi - ident)[0].cpu()
    #
    #     network_shape_list = list(ident.shape[2:])
    #     print(network_shape_list, "SOME network_shape_list")
    #     dimension = len(network_shape_list)
    #     print(dimension, "SOME DIMENSION")
    #     tr = itk.DisplacementFieldTransform[(itk.D, dimension)].New()
    #
    #     # We convert the displacement field into an itk Vector Image.
    #     scale = torch.Tensor(network_shape_list)
    #
    #     for _ in network_shape_list:
    #         scale = scale[:, None]
    #     disp *= scale - 1
    #
    #     # disp is a shape [3, H, W, D] tensor with vector components in the order [vi, vj, vk]
    #     disp_itk_format = (
    #         disp.double()
    #         .numpy()[list(reversed(range(dimension)))]
    #         .transpose(list(range(1, dimension + 1)) + [0])
    #     )
    #     # disp_itk_format is a shape [H, W, D, 3] array with vector components in the order [vk, vj, vi]
    #     # as expected by itk.
    #
    #     itk_disp_field = itk.image_from_array(disp_itk_format, is_vector=True)
    #
    #     tr.SetDisplacementField(itk_disp_field)
    #
    #     to_network_space = self.resampling_transform(image_A, list(reversed(network_shape_list)))
    #
    #     from_network_space = self.resampling_transform(
    #         image_B, list(reversed(network_shape_list))
    #     ).GetInverseTransform()
    #
    #     phi_AB_itk = itk.CompositeTransform[itk.D, dimension].New()
    #
    #     phi_AB_itk.PrependTransform(from_network_space)
    #     phi_AB_itk.PrependTransform(tr)
    #     phi_AB_itk.PrependTransform(to_network_space)
    #
    #     # warp(image_A, phi_AB_itk) is close to image_B
    #
    #     return phi_AB_itk
    #
    # def resampling_transform(self, image, shape):
    #
    #     imageType = itk.template(image)[0][itk.template(image)[1]]
    #
    #     dummy_image = itk.image_from_array(
    #         np.zeros(tuple(reversed(shape)), dtype=itk.array_from_image(image).dtype)
    #     )
    #     if len(shape) == 2:
    #         transformType = itk.MatrixOffsetTransformBase[itk.D, 2, 2]
    #     else:
    #         transformType = itk.VersorRigid3DTransform[itk.D]
    #     initType = itk.CenteredTransformInitializer[transformType, imageType, imageType]
    #     initializer = initType.New()
    #     initializer.SetFixedImage(dummy_image)
    #     initializer.SetMovingImage(image)
    #     transform = transformType.New()
    #
    #     initializer.SetTransform(transform)
    #     initializer.InitializeTransform()
    #
    #     if len(shape) == 3:
    #         transformType = itk.CenteredAffineTransform[itk.D, 3]
    #         t2 = transformType.New()
    #         t2.SetCenter(transform.GetCenter())
    #         t2.SetOffset(transform.GetOffset())
    #         transform = t2
    #     m = transform.GetMatrix()
    #     m_a = itk.array_from_matrix(m)
    #
    #     input_shape = image.GetLargestPossibleRegion().GetSize()
    #
    #     for i in range(len(shape)):
    #         m_a[i, i] = image.GetSpacing()[i] * (input_shape[i] / shape[i])
    #
    #     m_a = itk.array_from_matrix(image.GetDirection()) @ m_a
    #
    #     transform.SetMatrix(itk.matrix_from_array(m_a))
    #
    #     return transform
    # def model_step(self, batch: Any, is_3d=False):
    #     moving_img, fixed_img = batch
    #     # print(moving_img.shape)
    #     concat_img = torch.cat((moving_img, fixed_img), dim=1)
    #     print(concat_img.shape, "Shape before")
    #     # Define parameters for sliding window inference
    #     roi_size = (128, 128, 128)  # Adjust based on model and GPU memory
    #     sw_batch_size = 1  # Number of patches processed in parallel
    #     overlap = 0.4  # Overlap between patches
    #     map = self.netR_A.fullres_net.identity_map
    #     map = F.interpolate(map, size=(384, 320, 128), mode="trilinear",
    #                                         align_corners=False)
    #     if is_3d:
    #         moving_img_np = moving_img.cpu().detach().squeeze().numpy()
    #
    #         fixed_img_np = fixed_img.cpu().detach().squeeze().numpy()
    #
    #         moving_img_np = moving_img_np.transpose(2, 1, 0) # itk: D, W, H
    #         fixed_img_np = fixed_img_np.transpose(2, 1, 0)
    #
    #         moving_img_itk = itk.image_from_array(moving_img_np)
    #         fixed_img_itk = itk.image_from_array(fixed_img_np)
    #
    #         phi_AB, _, phi_AB_vecotorfield = register_pair(self.netR_A, moving_img_itk, fixed_img_itk)
    #         phi_AB_vecotorfield = F.interpolate(phi_AB_vecotorfield, size=(384, 320, 128), mode="trilinear", align_corners=False)
    #         concat_img = torch.cat((concat_img, map), dim=1)
    #         concat_img = torch.cat((concat_img, phi_AB_vecotorfield), dim=1)
    #         print(phi_AB_vecotorfield.shape, "Shape of the vector field")
    #         print(type(phi_AB_vecotorfield), "type of the vector field")
    #
    #         # Define the patch-based registration function for moving and fixed patches
    #         def _register_func(patch_pair):
    #             moving_patch, fixed_patch, map_patch, field_patch = patch_pair  # split moving and fixed patches
    #             print(moving_patch.shape, "shape of the moving patch")
    #             print(fixed_patch.shape, "shape of the fixed patch")
    #             print(map_patch.shape, "shape of the map patch")
    #             print(field_patch.shape, "shape of the field patch")
    #             moving_patch_np = moving_patch.cpu().detach().numpy().squeeze()
    #             fixed_patch_np = fixed_patch.cpu().detach().numpy().squeeze()
    #
    #
    #             # Check if the shape is as expected (Depth, Height, Width) for a 3D patch
    #             if moving_patch_np.ndim != 3:
    #                 raise ValueError("Moving patch shape is not 3-dimensional as expected.")
    #
    #             if fixed_patch_np.ndim != 3:
    #                 raise ValueError("Fixed patch shape is not 3-dimensional as expected.")
    #
    #             moving_img_itk = itk.image_from_array(moving_patch_np.transpose(2, 1, 0))
    #             fixed_img_itk = itk.image_from_array(fixed_patch_np.transpose(2, 1, 0))
    #             phi_AB_patch = self.create_itk_transform(field_patch, map_patch, moving_img_itk,
    #                                                      fixed_img_itk)
    #             # phi_AB, _ = register_pair(self.netR_A, moving_img_itk, fixed_img_itk)
    #
    #             interpolator = itk.LinearInterpolateImageFunction.New(moving_img_itk)
    #             warped_patch = itk.resample_image_filter(
    #                 moving_img_itk,
    #                 transform=phi_AB_patch,
    #                 interpolator=interpolator,
    #                 size=itk.size(fixed_img_itk),
    #                 output_spacing=itk.spacing(fixed_img_itk),
    #                 output_direction=fixed_img_itk.GetDirection(),
    #                 output_origin=fixed_img_itk.GetOrigin()
    #             )
    #
    #             # Convert the registered patch back to a tensor
    #             warped_patch_np = itk.array_from_image(warped_patch)
    #             warped_patch_tensor = torch.from_numpy(warped_patch_np).unsqueeze(0).unsqueeze(0)
    #             # print("Shape of the warped tensor patch", warped_patch_tensor.shape)
    #             return warped_patch_tensor.to(moving_patch.device)
    #
    #         # Prepare the patch pairs for inference
    #         def _patch_pair_inference(concat_patch):
    #             print(concat_patch.shape, "Shape after")
    #             return _register_func((concat_patch[:, 0], concat_patch[:, 1], concat_patch[:, 2:5], concat_patch[:, 5:]))
    #
    #         # Perform sliding window inference on both moving and fixed images
    #         warped_img = sliding_window_inference(
    #             concat_img, roi_size, sw_batch_size, _patch_pair_inference, overlap=overlap, mode="gaussian", sigma_scale=0.2
    #         )
    #         # warped_img = warped_img.transpose(2, 1, 0)
    #
    #         # print("Warped image final shape = ", warped_img.shape)
    #         # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    #         # axs[0].set_title("moving img")
    #         # axs[0].imshow(moving_img.squeeze().cpu().numpy()[..., 60], cmap="gray")
    #         #
    #         # axs[1].set_title("fixed img")
    #         # axs[1].imshow(fixed_img.squeeze().cpu().numpy()[..., 60], cmap="gray")
    #         #
    #         # axs[2].set_title("warped img")
    #         # axs[2].imshow(warped_img.squeeze().cpu().numpy()[..., 60], cmap="gray")
    #         # plt.show()
    #
    #         # Move the final output to the same device as moving_img
    #         warped_img = warped_img.to(moving_img.device)
    #         return moving_img, fixed_img, warped_img
    #
    #     else:
    #         print(concat_img.shape)
    #
    #         # Define the patch-based registration function for sliding window inference
    #         def _register_func2D(concat_patch):
    #             # Extract moving and fixed patches from concatenated patch
    #             moving_patch, fixed_patch = concat_patch
    #
    #             print(moving_patch.shape)
    #             print(fixed_patch.shape)
    #             # Perform registration on the patches
    #             loss, transform_vector, warped_patch = self.netR_A(moving_patch, fixed_patch)
    #
    #             return warped_patch
    #
    #         def _patch_pair(concat_image):
    #             return _register_func2D((concat_image[:, 0], concat_image[:, 1]))
    #
    #         warped_img = sliding_window_inference(
    #             concat_img, roi_size, sw_batch_size, _register_func2D, overlap=overlap
    #         )
    #         # Handle 2D or alternative cases if needed
    #         # loss, transform_vector, warped_img = self.netR_A(moving_img, fixed_img)
    #         fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    #         axs[0].set_title("moving img")
    #         axs[0].imshow(moving_img.squeeze().cpu().numpy(), cmap="gray")
    #
    #         axs[1].set_title("fixed img")
    #         axs[1].imshow(fixed_img.squeeze().cpu().numpy(), cmap="gray")
    #
    #         axs[2].set_title("warped img")
    #         axs[2].imshow(warped_img.squeeze().cpu().numpy(), cmap="gray")
    #         plt.show()
    #         return moving_img, fixed_img, warped_img


    #####################################################################################################################

    # def model_step(self, batch: Any, is_3d=False):
    #     moving_img, fixed_img = batch
    #     # print(moving_img.shape)
    #     concat_img = torch.cat((moving_img, fixed_img), dim=1)
    #     print(concat_img.shape, "Shape before")
    #     # Define parameters for sliding window inference
    #     roi_size = (128, 128, 128)  # Adjust based on model and GPU memory
    #     sw_batch_size = 1  # Number of patches processed in parallel
    #     overlap = 0.2  # Overlap between patches
    #
    #     if is_3d:
    #         # Define the patch-based registration function for moving and fixed patches
    #         def _register_func(patch_pair):
    #             moving_patch, fixed_patch = patch_pair  # split moving and fixed patches
    #             # print(f"Moving Patch Shape (moving_patch): {moving_patch.shape}")
    #             # print(f"Fixed Patch Shape (fixed_patch): {fixed_patch.shape}")
    #             # Convert patches to numpy and ITK for registration
    #             moving_patch_np = moving_patch.cpu().detach().numpy().squeeze()
    #             fixed_patch_np = fixed_patch.cpu().detach().numpy().squeeze()
    #             # Print shapes for debugging
    #             # print(f"Moving Patch Shape (before transpose): {moving_patch_np.shape}")
    #             # print(f"Fixed Patch Shape (before transpose): {fixed_patch_np.shape}")
    #             # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    #             # axs[0].set_title("moving patch")
    #             # axs[0].imshow(moving_patch_np[..., 60], cmap="gray")
    #             #
    #             # axs[1].set_title("fixed patch")
    #             # axs[1].imshow(fixed_patch_np[..., 60], cmap="gray")
    #             # plt.show()
    #
    #             # Check if the shape is as expected (Depth, Height, Width) for a 3D patch
    #             if moving_patch_np.ndim != 3:
    #                 raise ValueError("Moving patch shape is not 3-dimensional as expected.")
    #
    #             if fixed_patch_np.ndim != 3:
    #                 raise ValueError("Fixed patch shape is not 3-dimensional as expected.")
    #
    #
    #             moving_img_itk = itk.image_from_array(moving_patch_np.transpose(2, 1, 0))
    #             fixed_img_itk = itk.image_from_array(fixed_patch_np.transpose(2, 1, 0))
    #             # print(f"Moving Patch Shape (after transpose): {moving_img_itk.shape}")
    #             # print(f"Fixed Patch Shape (after transpose): {fixed_img_itk.shape}")
    #             # Register the moving patch to the fixed patch
    #             phi_AB, _ = register_pair(self.netR_A, moving_img_itk, fixed_img_itk)
    #             # displacement_field = None
    #             # for i in range(phi_AB.GetNumberOfTransforms()):
    #             #     transform = phi_AB.GetNthTransform(i)
    #             #     if isinstance(transform, itk.DisplacementFieldTransform):
    #             #         displacement_field = transform.GetDisplacementField()
    #             #         break
    #             #
    #             # if displacement_field is None:
    #             #     raise ValueError("No displacement field found in the composite transform")
    #             # sigma = [2.0, 2.0, 2.0]  # Adjust these values as needed
    #             # smoothed_field = itk.smooth_image_filter(displacement_field, sigma)
    #             # smoothed_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
    #             # smoothed_transform.SetDisplacementField(smoothed_field)
    #             # phi_AB.ClearTransformQueue()
    #             # Apply transformation and resample to obtain the registered patch
    #             interpolator = itk.LinearInterpolateImageFunction.New(moving_img_itk)
    #             warped_patch = itk.resample_image_filter(
    #                 moving_img_itk,
    #                 transform=phi_AB,
    #                 interpolator=interpolator,
    #                 size=itk.size(fixed_img_itk),
    #                 output_spacing=itk.spacing(fixed_img_itk),
    #                 output_direction=fixed_img_itk.GetDirection(),
    #                 output_origin=fixed_img_itk.GetOrigin()
    #             )
    #
    #             # Convert the registered patch back to a tensor
    #             warped_patch_np = itk.array_from_image(warped_patch)
    #             warped_patch_tensor = torch.from_numpy(warped_patch_np).unsqueeze(0).unsqueeze(0)
    #             # print("Shape of the warped tensor patch", warped_patch_tensor.shape)
    #             return warped_patch_tensor.to(moving_patch.device)
    #
    #         # Prepare the patch pairs for inference
    #         def _patch_pair_inference(concat_patch):
    #             print(concat_patch.shape, "Shape after")
    #             return _register_func((concat_patch[:, 0], concat_patch[:, 1]))
    #
    #         # Perform sliding window inference on both moving and fixed images
    #         warped_img = sliding_window_inference(
    #             concat_img, roi_size, sw_batch_size, _patch_pair_inference, overlap=overlap
    #         )
    #         # warped_img = warped_img.transpose(2, 1, 0)
    #
    #         # print("Warped image final shape = ", warped_img.shape)
    #         # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    #         # axs[0].set_title("moving img")
    #         # axs[0].imshow(moving_img.squeeze().cpu().numpy()[..., 60], cmap="gray")
    #         #
    #         # axs[1].set_title("fixed img")
    #         # axs[1].imshow(fixed_img.squeeze().cpu().numpy()[..., 60], cmap="gray")
    #         #
    #         # axs[2].set_title("warped img")
    #         # axs[2].imshow(warped_img.squeeze().cpu().numpy()[..., 60], cmap="gray")
    #         # plt.show()
    #
    #         # Move the final output to the same device as moving_img
    #         warped_img = warped_img.to(moving_img.device)
    #         return moving_img, fixed_img, warped_img
    #
    #     else:
    #         print(concat_img.shape)
    #         # Define the patch-based registration function for sliding window inference
    #         def _register_func2D(concat_patch):
    #             # Extract moving and fixed patches from concatenated patch
    #             moving_patch, fixed_patch = concat_patch
    #
    #             print(moving_patch.shape)
    #             print(fixed_patch.shape)
    #             # Perform registration on the patches
    #             loss, transform_vector, warped_patch = self.netR_A(moving_patch, fixed_patch)
    #
    #             return warped_patch
    #
    #         def _patch_pair(concat_image):
    #             return _register_func2D((concat_image[:, 0], concat_image[:, 1]))
    #
    #         warped_img = sliding_window_inference(
    #             concat_img, roi_size, sw_batch_size, _register_func2D, overlap=overlap
    #         )
    #         # Handle 2D or alternative cases if needed
    #         # loss, transform_vector, warped_img = self.netR_A(moving_img, fixed_img)
    #         fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    #         axs[0].set_title("moving img")
    #         axs[0].imshow(moving_img.squeeze().cpu().numpy(), cmap="gray")
    #
    #         axs[1].set_title("fixed img")
    #         axs[1].imshow(fixed_img.squeeze().cpu().numpy(), cmap="gray")
    #
    #         axs[2].set_title("warped img")
    #         axs[2].imshow(warped_img.squeeze().cpu().numpy(), cmap="gray")
    #         plt.show()
    #         return moving_img, fixed_img, warped_img

    # def model_step(self, batch: Any, is_3d=False):
    #     moving_img, fixed_img = batch
    #
    #     if is_3d:
    #         moving_img_np = moving_img.cpu().detach().squeeze().numpy()
    #         fixed_img_np = fixed_img.cpu().detach().squeeze().numpy()
    #         print(moving_img_np.shape)
    #         # Dimensions
    #         H, W, D = moving_img_np.shape
    #         d_w, w_w, h_w = (128, 128, 128)
    #         d_o, w_o, h_o = (32, 32, 32)
    #         print(D, W, H)
    #         # Initialize an empty array to store the reconstructed warped image
    #         warped_img_np = np.zeros_like(moving_img_np)
    #         count_map = np.zeros_like(moving_img_np)  # To keep track of overlaps for averaging
    #
    #         # Iterate over the volume using a sliding window
    #         for d in range(0, D - d_w + 1, d_w - d_o):
    #             for w in range(0, W - w_w + 1, w_w - w_o):
    #                 for h in range(0, H - h_w + 1, h_w - h_o):
    #                     # Adjust end indices to stay within bounds
    #                     d_end = min(d + d_w, D)
    #                     w_end = min(w + w_w, W)
    #                     h_end = min(h + h_w, H)
    #                     # Extract patches
    #                     moving_patch = moving_img_np[h:h_end, w:w_end, d:d_end]
    #                     fixed_patch = fixed_img_np[h:h_end, w:w_end, d:d_end]
    #                     print(moving_patch.shape)
    #                     print(fixed_patch.shape)
    #
    #                     # Convert patches to ITK images
    #                     moving_img_itk = itk.image_from_array(moving_patch)
    #                     fixed_img_itk = itk.image_from_array(fixed_patch)
    #
    #                     # Perform registration on the patch
    #                     phi_AB, phi_BA = register_pair(self.netR_A, moving_img_itk, fixed_img_itk)
    #                     interpolator = itk.LinearInterpolateImageFunction.New(moving_img_itk)
    #                     warped_patch = itk.resample_image_filter(
    #                         moving_img_itk,
    #                         transform=phi_AB,
    #                         interpolator=interpolator,
    #                         size=itk.size(fixed_img_itk),
    #                         output_spacing=itk.spacing(fixed_img_itk),
    #                         output_direction=fixed_img_itk.GetDirection(),
    #                         output_origin=fixed_img_itk.GetOrigin()
    #                     )
    #                     warped_patch_np = itk.array_from_image(warped_patch)
    #
    #                     # Insert the warped patch into the output image
    #                     warped_img_np[h:h + h_w, w:w + w_w, d:d + d_w] += warped_patch_np
    #                     count_map[h:h + h_w, w:w + w_w, d:d + d_w] += 1
    #
    #         # Normalize the overlapping regions by dividing by the count map
    #         warped_img_np = warped_img_np / np.maximum(count_map, 1)  # Avoid division by zero
    #
    #         # Convert back to a torch tensor
    #         warped_img_tensor = torch.from_numpy(warped_img_np).unsqueeze(0).unsqueeze(0)
    #         warped_img_tensor = warped_img_tensor.to(moving_img.device)
    #
    #         return moving_img, fixed_img, warped_img_tensor
    #
    #     else:
    #         # original_slices = evaluation_img.shape[-1] #Padding is not needed
    #         # moving_img_pad = self.pad_slice_to_128(moving_img)
    #         # fixed_img_pad = self.pad_slice_to_128(fixed_img)
    #         # loss, transform_vector, warped_img_pad = self.netR_A(moving_img_pad, fixed_img_pad)
    #         loss, transform_vector, warped_img = self.netR_A(moving_img, fixed_img)
    #         # warped_img = self.crop_slice_to_original(warped_img_pad, original_slices)
    #         return moving_img, fixed_img, warped_img


        
    def training_step(self, batch: Any, batch_idx: int):
        # print("HI")
        # start = time.time()
        optimizer_R_A = self.optimizers()
        
        with optimizer_R_A.toggle_model():
            loss = self.model_step_for_train(batch)
            self.manual_backward(loss)
            self.clip_gradients(
                optimizer_R_A, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
            )
            optimizer_R_A.step()
            optimizer_R_A.zero_grad()

        self.log("loss", loss.detach(), prog_bar=True)
        # end = time.time()
        # print("Step took: ", end - start)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizers = []
        schedulers = []
        
        optimizer_R_A = self.hparams.optimizer(params=self.netR_A.parameters())
        optimizers.append(optimizer_R_A)

        if self.hparams.scheduler is not None:
            scheduler_R_A = self.hparams.scheduler(optimizer=optimizer_R_A)
            schedulers.append(scheduler_R_A)
            return optimizers, schedulers
        
        return optimizers