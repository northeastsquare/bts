import math
import numpy as np
import numpy.ma as ma
import cv2, os
import time

if __name__ == '__main__':
	sum_rmse = 0
	count = 0
	avg_rmse = 0
	pred_path = '/home/hello/work/bts/result_bts_eigen/raw'
	gt_depth_fns = []
	pred_fnames = []
	with open('./train_test_inputs/rili_test_files_with_gt.txt') as f:
		lines = f.readlines()
		for l in lines:
			items = l.strip().split(' ')
			#pred_filenames.append(items[0].strip())
			gfn = items[1].strip()
			fn = gfn[gfn.rfind('/')+1:]
			
			pred_fn = os.path.join(pred_path, fn)
			pred_fnames.append(pred_fn)
			gt_depth_fns.append(gfn)


	"""
	im_fnames = []
	for root, dirs, files in os.walk(pred_path, topdown=True):
		for name in files:
			fn, ext = os.path.splitext(name)
			if ext != '.png':
				continue
			pred_fnames.append(os.path.join(root,name))
			imfn = os.path.join(root[:root.rfind('/')], 'rgb', fn.replace('Depth', 'Color')+'.jpg')
			im_fnames.append(imfn)"""
	#print("file len:", len(im_fnames))
	for i, pred_fn in enumerate(pred_fnames):
		gt_fn = gt_depth_fns[i]
		print("fn:", gt_fn, pred_fn)
		groundtruth_image = cv2.imread(gt_fn,2)
		calculated_image = cv2.imread(pred_fn, 2)
		if (groundtruth_image is not None) and (calculated_image is not None):
			if (groundtruth_image.shape[0]==calculated_image.shape[0]) and(groundtruth_image.shape[1]==calculated_image.shape[1]):
				groundtruth_image = groundtruth_image / 1000.0  # mm->m
				calculated_image = calculated_image / 1000.0
				# rmse calculation#
				mask = (groundtruth_image > 20.0) | (groundtruth_image <= 0)
				mx = ma.array(groundtruth_image, mask=mask)
				mz = ma.array(calculated_image, mask=mask)
				tmp_rmse = np.sqrt(np.mean((mx - mz) ** 2))
				# rmse calculation#
				sum_rmse = avg_rmse * count + tmp_rmse
				count = count + 1
				avg_rmse = sum_rmse / count
				
				print(' count: ', count, ' current_rmse: ', round(tmp_rmse, 3), ' avg_rmse: ',
						round(avg_rmse, 3))
				depthimage_colormap = cv2.applyColorMap(cv2.convertScaleAbs(calculated_image, alpha=30),
														cv2.COLORMAP_JET)
													
				cv2.imshow('depth', depthimage_colormap)
				cv2.waitKey(0)
		
		#print('cost time: ',time.time()-start_t)
