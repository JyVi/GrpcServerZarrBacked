import json
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.affines import apply_affine
from skimage.morphology import skeletonize
from skimage.morphology import thin
nii    = nib.load("/Users/jlepinay/SPIMED/DataStorage/ImageCAS_Batch1_Stack0/scan.nii")
data   = nii.get_fdata()
niiCoro = nib.load("/Users/jlepinay/SPIMED/DataStorage/ImageCAS_Batch1_Stack0/coroBinMask01.nii")
dataCoro = niiCoro.get_fdata()
print(np.unique(dataCoro))
print(dataCoro.shape)
print(np.count_nonzero(dataCoro))

skel = skeletonize(dataCoro)
print(skel.shape)
print("Skeleton unique values:", np.unique(skel))
print("Skeleton non-zero count:", np.count_nonzero(skel))