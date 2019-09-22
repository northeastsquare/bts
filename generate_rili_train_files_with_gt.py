import os
imdir = '/home/hello/dataset_rili_correction/'
tftxt =  open('train_test_inputs/rili_train_files_with_gt.txt', 'w')
for root, dirs, files in os.walk(imdir, topdown=True):
    for name in files:
        fn, ext = os.path.splitext(name)
        if ext != '.jpg':
            continue
        #Depth, Color
        depth_color, sid = fn.split('_')
        print(root, "name:", name, depth_color, sid)
        if depth_color != 'Color':
            continue
        parent_root = root[:root.rfind('/')]
        
        depth_path = os.path.join(parent_root, 'depth', 'Depth_'+sid+'.png')
        if not os.path.exists(depth_path):
            continue
        image_path = os.path.join(root, name)
        stowrite = image_path + ' ' + depth_path + ' 597.267'+'\n'
        tftxt.write(stowrite)        
