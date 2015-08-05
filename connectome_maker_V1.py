#!/usr/bin/env python

'''
 The purpose of this script is to integrate a series of dipy functions into one coherent operations, to give 
 you a structural connectome output for each subject.
 Before you run this script, its important that you create a coregistered atlas (coregistered to your b0 scan)
 Obviously, most of the code (and commentary) below is straight from dipy, e.g. 
 http://nipy.org/dipy/examples_built/streamline_tools.html. See the original website for more complete
 commentary. 

 Also uses a helper script (condition_seeds.py) to resolve nonzero elements.
'''

subnums = [  'subject01' ] # subject list goes here.

for subnum in subnums:

	'''
	Import all of the important libraries and initialize data files. Might be unnecessary given your setup.
	'''
	import sys
	sys.path.append('/imaging/local/software/python_packages/dipy/0.8.0/lib.linux-x86_64-2.7')
	import dipy
	sys.path.append('/imaging/local/software/python_packages/nibabel/1.3.0')
	import nibabel

	import numpy as np
	import nibabel as nib
	from os.path import join
	from dipy.reconst.dti import TensorModel, fractional_anisotropy
	from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel, auto_response)
	from dipy.direction import peaks_from_model
	from dipy.data import get_sphere
	from dipy.io import read_bvals_bvecs
	from dipy.core.gradients import gradient_table
	from dipy.tracking.eudx import EuDX
	from dipy.reconst import peaks, shm, recspeed
	from dipy.reconst.dti import TensorModel, fractional_anisotropy
	from dipy.tracking import utils
	from dipy.segment.mask import median_otsu
	from condition_seeds import condition_seeds

	# point to the diffusion data
	home = expanduser('~')
	dname = ('/imaging/diffusion_data_location/' + str(subnum) + '/diffusion')
	fdwi = join(dname, '_etc.nii')
	img = nib.load(fdwi)
	data = img.get_data()

	# point to the bvecs and bvals files
	bname = ('/imaging/diffusion_data_location/' + str(subnum) + '/diffusion')
	fbval = join(bname, 'bvals')
	fbvec = join(bname, 'bvecs')

	bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
	gtab = gradient_table(bvals, bvecs)

	# this line can be changed to reflect different atlasses; you just need to coregister atlas to native space.
	atlas_dir = ('/imaging/subject_HOAs')
	atlas_file = join(atlas_dir, (str(subnum) +'_hoa_thr_LR.nii.gz'))
	atlas = nib.load(atlas_file)
	labels = atlas.get_data()

	# mask the data, creates a mask '''
	print '\tCreating Mask'
	data_masked, mask = median_otsu(data, 2, 1)

	'''
	We've loaded an image called `atlas` which is a map of tissue types such
	that every integer value in the array `labels` represents an anatomical
	structure or tissue type [#]_.  We'll use `peaks_from_model` to apply the 
	`CsaOdfModel` to each white matter voxel and estimate fiber orientations which 
	we can use for tracking.
	'''

	print '\tCalculating peaks'
	csamodel = shm.CsaOdfModel(gtab, 6)
	csapeaks = peaks.peaks_from_model(model=csamodel,
	                                  data=data,
	                                  sphere=peaks.default_sphere,
	                                  relative_peak_threshold=.5,
	                                  min_separation_angle=25,
	                                  mask=mask)

	'''
	Brief interlude to make sure we don't seed from low-FA voxels.
	'''

	print '\tTensor Fitting'
	tensor_model = TensorModel(gtab, fit_method='WLS')
	tensor_fit = tensor_model.fit(data, mask)

	FA = fractional_anisotropy(tensor_fit.evals)
	stopping_values = np.zeros(csapeaks.peak_values.shape)
	stopping_values[:] = FA[..., None]

	'''
	Now we can use EuDX to track all of the white matter. To keep things reasonably
	fast we use `density=2` which will result in 8 seeds per voxel. We'll set
	`a_low` (the parameter which determines the threshold of FA/QA under which
	tracking stops) to be very low because we've already applied a white matter
	mask.
	'''
	print '\tTracking'
	seeds = utils.seeds_from_mask(mask, density=2)
	condition_seeds = condition_seeds(seeds, np.eye(4),
	csapeaks.peak_values.shape[:3], verbose=1)   # Here's the line for 
	streamline_generator = EuDX(stopping_values, csapeaks.peak_indices,
	                            odf_vertices=peaks.default_sphere.vertices,
	                            a_low=.05, step_sz=.5, seeds=condition_seeds)
	affine = streamline_generator.affine
	streamlines = list(streamline_generator)

	'''
	Streamlines are a path though the 3d space of an image represented by a
	set of points. For these points to have a meaningful interpretation, these
	points must be given in a known coordinate system. The ``affine`` attribute of
	the ``streamline_generator`` object specifies the coordinate system of the
	points with respect to the voxel indices of the input data.

	Next we want to create a matrix of streamline connections. To do this we can use
	the `connectivity_matrix` function. This function takes a set of streamlines
	and an array of labels as arguments. It returns the number of streamlines that
	start and end at each pair of labels and it can return the streamlines grouped
	by their endpoints. Notice that this function only considers the endpoints of
	each streamline.
	'''

	M, grouping = utils.connectivity_matrix(streamlines, labels, affine=affine,
	                                      return_mapping=True,
	                                      mapping_as_streamlines=True)


	'''
	We've set ``return_mapping`` and ``mapping_as_streamlines`` to ``True`` so that
	``connectivity_matrix`` returns all the streamlines in ``cc_streamlines``
	grouped by their endpoint.

	Save the matrix into a csv file.
	'''

	np.savetxt((str(subnum) + '_connectome.csv'), M, delimiter=',')
	print('Finished with subject ' + str(subnum))

	'''
	Other things to do:
	- remove first row/column
	- adjust each streamline count by the total area in the seed and target ROIs
	- output tract maps for use in FA mapping
	'''
