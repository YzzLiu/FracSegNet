import vtkmodules.all as vtk
import SimpleITK as sitk
import numpy as np
import os


def ITKImage2VTKImage_no_Origin(image0):
    imageArray = sitk.GetArrayFromImage(image0)
    dataImporter = vtk.vtkImageImport()
    if (type(imageArray[0][0][0]) == type(np.int16())):
        dataImporter.SetDataScalarTypeToShort()
    elif (type(imageArray[0][0][0]) == type(np.int32())):
        dataImporter.SetDataScalarTypeToInt()
    elif (type(imageArray[0][0][0]) == type(np.uint16())):
        dataImporter.SetDataScalarTypeToUnsignedShort()
    elif (type(imageArray[0][0][0]) == type(np.uint8())):
        dataImporter.SetDataScalarTypeToUnsignedChar()
    elif (type(imageArray[0][0][0]) == type(np.float())):
        dataImporter.SetDataScalarTypeToFloat()
    elif (type(imageArray[0][0][0]) == type(np.double())):
        dataImporter.SetDataScalarTypeToDouble()
    else:
        return None
    # arrBytes = imageArray.tobytes()
    # dataImporter.CopyImportVoidPointer(imageArray, imageArray.size)
    dataImporter.SetImportVoidPointer(imageArray)
    spacing = image0.GetSpacing()
    # origin=image0.GetOrigin()
    dim = image0.GetSize()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetWholeExtent(0, dim[0] - 1, 0, dim[1] - 1, 0, dim[2] - 1)
    dataImporter.SetDataExtentToWholeExtent()
    dataImporter.SetDataSpacing(spacing[0], spacing[1], spacing[2])
    # dataImporter.SetDataOrigin(origin[0], origin[1], origin[2])
    dataImporter.Update()
    vtkImage = vtk.vtkImageData()
    vtkImage.DeepCopy(dataImporter.GetOutput())
    return vtkImage


def remove_slices_from_image(img_itk):
    img_arr = sitk.GetArrayFromImage(img_itk)
    n = np.int16(img_arr.shape[0] * 0.085)
    img_arr[:n] = 0
    img_arr[-n:] = 0
    img_new = sitk.GetImageFromArray(img_arr)
    img_new.SetDirection(img_itk.GetDirection())
    img_new.SetOrigin(img_itk.GetOrigin())
    img_new.SetSpacing(img_itk.GetSpacing())

    return img_new


def save_STL_from_ITKImage(img_itk, dir_stl=None, smooth_iter=20, rm_slices=True):
    if rm_slices:
        img_itk = remove_slices_from_image(img_itk)

    img_vtk = ITKImage2VTKImage_no_Origin(img_itk)
    # compute the surface mesh
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputData(img_vtk)
    surf.SetValue(0, 1)  # use surf.GenerateValues function if more than one contour is available in the file
    surf.Update()

    # smoothing the mesh
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(surf.GetOutputPort())
    # increase this integer set number of iterations if smoother surface wanted
    smoother.SetNumberOfIterations(smooth_iter)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOff()  # The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
    smoother.GenerateErrorScalarsOn()
    smoother.Update()

    # taking into account image direction and origin by applying a transfrom
    m33 = np.array(img_itk.GetDirection()).reshape(3, 3)
    m13 = np.array(img_itk.GetOrigin()).reshape(1, 3)
    m44 = np.identity(4)
    m44[:3, :3] = m33
    m44[:3, 3] = m13
    m = vtk.vtkMatrix4x4()
    m.DeepCopy(m44.flatten())
    transform = vtk.vtkTransform()
    transform.SetMatrix(m)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(smoother.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    # clipper=vtkClipClosedSurface()
    # clipper.SetInputConnection(transform_filter.GetOutputPort())

    # save
    if dir_stl == None:
        dir_stl = 'temp.stl'
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(transform_filter.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(dir_stl)
    writer.Write()
    print("STL file saved:", dir_stl)
    return dir_stl


def nii2stl(dir_nii, dir_stl=None, smooth_iter=20):
    # vtk read nii
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(dir_nii)
    reader.Update()

    # compute the surface mesh
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    surf.SetValue(0, 1)  # use surf.GenerateValues function if more than one contour is available in the file
    surf.Update()

    # smoothing the mesh
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(surf.GetOutputPort())
    # increase this integer set number of iterations if smoother surface wanted
    smoother.SetNumberOfIterations(smooth_iter)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOff()  # The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
    smoother.GenerateErrorScalarsOn()
    smoother.Update()

    # taking into account image direction and origin by applying a transfrom
    img_itk = sitk.ReadImage(dir_nii)
    m33 = np.array(img_itk.GetDirection()).reshape(3, 3)
    m13 = np.array(img_itk.GetOrigin()).reshape(1, 3)
    m44 = np.identity(4)
    m44[:3, :3] = m33
    m44[:3, 3] = m13
    m = vtk.vtkMatrix4x4()
    m.DeepCopy(m44.flatten())
    transform = vtk.vtkTransform()
    transform.SetMatrix(m)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(smoother.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    # clipper=vtkClipClosedSurface()
    # clipper.SetInputConnection(transform_filter.GetOutputPort())

    # save
    if dir_stl == None:
        dir_stl = dir_nii.replace('.nii.gz', '.stl')
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(transform_filter.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(dir_stl)
    writer.Write()
    return dir_stl


def nii2stl_folder(data_path):
    for a in os.listdir(data_path):
        if ".nii.gz" in a:
            nii2stl(os.path.join(data_path, a))
