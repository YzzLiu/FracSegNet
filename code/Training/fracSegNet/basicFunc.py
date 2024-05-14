import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def boneData(Vdir,maskdir,label):
    # 0.load
    V = sitk.ReadImage(Vdir)
    mask =  sitk.ReadImage(maskdir)
    # 1. scale
    img_T1 = sitk.Cast(sitk.RescaleIntensity(V), sitk.sitkUInt8)
    # 2. extract pelvic
    itk_skeleton = GetLabelImage(img_T1, mask,label)
    itk_skeleton_scale = sitk.Cast(sitk.RescaleIntensity(itk_skeleton), sitk.sitkUInt8)
    # resample
    itk_skeleton_reshape = ImageResample(itk_skeleton_scale)
    # 3. basic params
    itk_skeleton_reshape.SetOrigin=[0,0,0]
    showParams(itk_skeleton_reshape)
    # 4. show fusion image
    #sitk.Show(sitk.LabelOverlay(itk_skeleton_scale, mask))
    #sitk.Show(itk_skeleton_scale)
    return itk_skeleton_reshape

def pelvisData(Vdir,maskdir):
    # 0.load
    V = sitk.ReadImage(Vdir)
    mask =  sitk.ReadImage(maskdir)
    # 1. scale
    img_T1 = sitk.Cast(sitk.RescaleIntensity(V), sitk.sitkUInt8)
    # 2. extract pelvic
    itk_skeleton = GetMaskImage(img_T1, mask)
    itk_skeleton_scale = sitk.Cast(sitk.RescaleIntensity(itk_skeleton), sitk.sitkUInt8)
    # resample
    itk_skeleton_reshape = ImageResample(itk_skeleton_scale)
    # 3. basic params
    # itk_skeleton_reshape.SetOrigin=[0,0,0]
    showParams(itk_skeleton_reshape)
    # 4. show fusion image
    #sitk.Show(sitk.LabelOverlay(itk_skeleton_scale, mask))
    #sitk.Show(itk_skeleton_scale)
    return itk_skeleton_reshape

def pelvisOriginData(Vdir,maskdir):
    # 0.load
    img_T1 = sitk.ReadImage(Vdir)
    mask =  sitk.ReadImage(maskdir)
    # 1. scale
    # img_T1 = sitk.Cast(sitk.RescaleIntensity(V), sitk.sitkUInt8)
    # 2. extract pelvic
    itk_skeleton_scale = GetMaskImage(img_T1, mask)
    # itk_skeleton_scale = sitk.Cast(sitk.RescaleIntensity(itk_skeleton), sitk.sitkUInt8)
    # resample
    itk_skeleton_reshape = ImageResample(itk_skeleton_scale)
    # 3. basic params
    # itk_skeleton_reshape.SetOrigin=[0,0,0]
    showParams(itk_skeleton_reshape)
    # 4. show fusion image
    #sitk.Show(sitk.LabelOverlay(itk_skeleton_scale, mask))
    #sitk.Show(itk_skeleton_scale)
    return itk_skeleton_reshape

def GetMaskImage(sitk_src, sitk_mask, replacevalue=0):
    array_src = sitk.GetArrayFromImage(sitk_src)
    array_mask = sitk.GetArrayFromImage(sitk_mask)
    array_out = array_src.copy()
    array_out[array_mask == 0] = replacevalue
    outmask_sitk = sitk.GetImageFromArray(array_out)
    outmask_sitk.SetDirection(sitk_src.GetDirection())
    outmask_sitk.SetSpacing(sitk_src.GetSpacing())
    outmask_sitk.SetOrigin(sitk_src.GetOrigin())
    return outmask_sitk

def GetLabelImage(sitk_src, sitk_mask, label ,replacevalue=0):
    array_src = sitk.GetArrayFromImage(sitk_src)
    array_mask = sitk.GetArrayFromImage(sitk_mask)
    array_out = array_src.copy()
    array_out[array_mask != label] = replacevalue
    if label == 0:
        array_out[array_mask == label] = replacevalue
    outmask_sitk = sitk.GetImageFromArray(array_out)
    outmask_sitk.SetDirection(sitk_src.GetDirection())
    outmask_sitk.SetSpacing(sitk_src.GetSpacing())
    outmask_sitk.SetOrigin(sitk_src.GetOrigin())
    return outmask_sitk

def showParams(itk_skeleton):
    print('size:',itk_skeleton.GetSize())
    print('spacing:',itk_skeleton.GetSpacing())
    print('origin:',itk_skeleton.GetOrigin())
    print('Direction:',itk_skeleton.GetDirection())
    print('PixelType:',itk_skeleton.GetPixelIDTypeAsString())

def ImageResample(sitk_image, new_spacing = [1.0, 1.0, 1.0], is_label = False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        #resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage

# input: ct_origin_fileName,ct_label_fileName,label output:frac_Grayscale==label
def extractSingleFrac(ct_scale_img, ct_label_img, label):
    # ct_origin_img = sitk.ReadImage(ct_origin_fileName)
    # ct_label_img = sitk.ReadImage(ct_label_fileName)
    ct_origin_arr = sitk.GetArrayFromImage(ct_scale_img)  # get array from image
    ct_label_arr = sitk.GetArrayFromImage(ct_label_img)  # get array from image
    frac_Grayscale = ct_origin_arr.copy()
    frac_img_norm = ct_origin_arr.copy()
    frac_Grayscale[ct_label_arr != label] = 0
    # frac_img_norm[ct_label_arr == label] = 1+((frac_img_norm[ct_label_arr == label]-min(frac_img_norm))*255/(max(frac_img_norm)-min(frac_img_norm)))
    frac_img_norm[ct_label_arr != label] = np.min(frac_Grayscale) - 1
    # print(np.min(frac_Grayscale))

    frac_img_norm = sitk.GetImageFromArray(frac_img_norm)
    frac_img_norm.SetDirection(ct_scale_img.GetDirection())
    frac_img_norm.SetSpacing(ct_scale_img.GetSpacing())
    frac_img_norm.SetOrigin(ct_scale_img.GetOrigin())

    frac_Grayscale = sitk.GetImageFromArray(frac_Grayscale)
    frac_Grayscale.SetDirection(ct_scale_img.GetDirection())
    frac_Grayscale.SetSpacing(ct_scale_img.GetSpacing())
    frac_Grayscale.SetOrigin(ct_scale_img.GetOrigin())

    return frac_Grayscale, frac_img_norm
# - ---------------------------------------------------------------------image show ------------------------------------------------
# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1, 2, figsize=(10, 8))

    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_npa[fixed_image_z, :, :], cmap=plt.cm.Greys_r)
    plt.title("fixed image")
    plt.axis("off")

    # Draw the moving image in the second subplot.
    plt.subplot(1, 2, 2)
    plt.imshow(moving_npa[moving_image_z, :, :], cmap=plt.cm.Greys_r)
    plt.title("moving image")
    plt.axis("off")

    plt.show()

