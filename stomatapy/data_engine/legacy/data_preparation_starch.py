import numpy as np # NumPy
import shutil # for copy files
import random # generate random numbers
from PIL import Image # Pillow image processing
import albumentations as A # for data augmentation
from matplotlib import pyplot as plt # show images and plot figures
from .. core.load_images import * # get our core functions

crop_size = 512 # image width and height

gc_color = [GC_Colors[idx].mask_rgb for idx in range(len(GC_Colors))] # GC colors list
starch_color = [Starch_Colors[idx].mask_rgb for idx in range(len(Starch_Colors))] # Starch colors list
starch_color.append([255, 255, 255]) # add cell wall

def quality_check(images_dir, masks_GC_dir, masks_Starch_dir, check=True):
    """check masks quality """
    if check:
        # try to remove .DS_Store files
        try:
            os.remove(os.path.join(images_dir, '.DS_Store'))
            os.remove(os.path.join(masks_GC_dir, '.DS_Store'))
            os.remove(os.path.join(masks_Starch_dir, '.DS_Store'))
        except BaseException:
            pass

        # check if images and masks match
        if set(os.listdir(images_dir)) != set(os.listdir(masks_GC_dir)):
            print('files in Images and Masks_Starch do not match')
        if set(os.listdir(images_dir)) != set(os.listdir(masks_Starch_dir)):
            print('files in Images and Masks_GC do not match')

    # load images and masks
    file_names = sorted(os.listdir(images_dir), key=str.casefold)
    file_names = [name for name in file_names if any(name.endswith(file_type) for file_type in image_types)]
    images = [imread_rgb(os.path.join(images_dir, file)) for file in file_names]
    masks_GC = [imread_rgb(os.path.join(masks_GC_dir, file)) for file in file_names]
    masks_Starch = [imread_rgb(os.path.join(masks_Starch_dir, file)) for file in file_names]

    # check if masks contain non-class value
    if check:      
        for idx, name in enumerate(file_names):
            if images[idx].shape != (512, 512, 3) or images[idx].shape != masks_GC[idx].shape or images[idx].shape != masks_Starch[idx].shape:
                print(f'{name} dimension is not 512x512x3')
            if not np.array_equal(unique_color(masks_GC[idx]), gc_color) or not np.isin(unique_color(masks_Starch[idx]), starch_color).all():
                print(f'{name} mask contains non-class value')
    return images, masks_GC, masks_Starch, file_names


def data_augmentation(image, mask_GC, mask_Starch, alpha=500, sigma=20, show_results=False):
    augmentations = A.Compose([
        A.PadIfNeeded(min_height=crop_size*2, min_width=crop_size*2, border_mode=cv2.BORDER_REPLICATE, always_apply=True),
        A.Flip(always_apply=True), # flip the input either horizontally, vertically or both
        A.Rotate(limit=(-180, 180), interpolation=cv2.INTER_LANCZOS4, always_apply=True), # rotate the input by an angle selected randomly from the uniform distribution.
        A.RandomScale(scale_limit=0.1, interpolation=cv2.INTER_LANCZOS4, always_apply=True), # randomly resize the input
        A.ElasticTransform (alpha=alpha, sigma=sigma, interpolation=cv2.INTER_LANCZOS4, always_apply=True), # elastic deformation of images
        A.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.2, always_apply=True), # randomly change the brightness, contrast, and saturation of an image
        A.CenterCrop(height=crop_size, width=crop_size, always_apply=True), # back to original size
        ],
        additional_targets={'mask_GC': 'image', 'mask_Starch': 'image'})

    augmentations.additional_targets['mask_GC'] = 'mask'
    augmentations.additional_targets['mask_Starch'] = 'mask'
    augmented = augmentations(image=image, mask_GC=mask_GC, mask_Starch=mask_Starch) # apply augmentation

    if show_results:
        figure, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4)) # plot layout 1 row 3 images
        ax1.imshow(image); ax1.set_title('original image'); ax1.axis('off') # original image
        ax2.imshow(augmented['image']); ax2.set_title('augmented'); ax2.axis('off') # original augmented
        ax3.imshow(augmented['mask_GC'], cmap=plt.cm.gray); ax3.set_title('GC mask'); ax3.axis('off') # GC mask augmented
        ax4.imshow(augmented['mask_Starch'], cmap=plt.cm.gray); ax4.set_title('Starch mask'); ax4.axis('off') # Starch mask augmented

    if not np.array_equal(unique_color(augmented['mask_GC']), gc_color) or not np.isin(unique_color(augmented['mask_Starch']), starch_color).all():
        print('data augmentation changed mask RGB value!')
    return augmented['image'], augmented['mask_GC'], augmented['mask_Starch']


def augmentation_creator(images_dir, masks_GC_dir, masks_Starch_dir, alpha=500, sigma=20, n_augmentation=4):
    """create augmented images """
    aug_images_dir = os.path.join(images_dir.rsplit('//', 1)[:-1][0], 'Aug_Images')
    aug_masks_GC_dir = os.path.join(masks_GC_dir.rsplit('//', 1)[:-1][0], 'Aug_masks_GC')
    aug_masks_Starch_dir = os.path.join(masks_Starch_dir.rsplit('//', 1)[:-1][0], 'Aug_masks_Starch')
    os.makedirs(aug_images_dir, exist_ok=True) # new dir for augmented images
    os.makedirs(aug_masks_GC_dir, exist_ok=True) # new dir for augmented GC masks
    os.makedirs(aug_masks_Starch_dir, exist_ok=True) # new dir for augmented Starch masks

    images, masks_GC, masks_Starch, file_names = quality_check(images_dir, masks_GC_dir, masks_Starch_dir, check=False)

    for idx, name in tqdm(enumerate(file_names), total=len(file_names)):
        image =  images[idx] # load original image
        mask_GC = masks_GC[idx] # load guard cell mask
        mask_Starch = masks_Starch[idx] # load starch mask
        mask_Starch = color_select(mask_Starch, mask_Starch, Starch_Colors[1].mask_rgb) # erase cell wall (white)
        cv2.imwrite(os.path.join(aug_images_dir, name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) # copy original to new dir
        cv2.imwrite(os.path.join(aug_masks_GC_dir, name), cv2.cvtColor(mask_GC, cv2.COLOR_RGB2BGR)) # copy GC mask to new dir
        cv2.imwrite(os.path.join(aug_masks_Starch_dir, name), cv2.cvtColor(mask_Starch, cv2.COLOR_RGB2BGR)) # copy starch mask to new dir
        for i in range(n_augmentation):
            # apply augmentation n times for each image
            new_name = os.path.splitext(file_names[idx])[0] + ' Aug_' + str(i) + os.path.splitext(file_names[idx])[1] # e.g. 'file aug_i.tif'
            aug_image, aug_mask_GC, aug_mask_Starch = data_augmentation(image, mask_GC, mask_Starch, alpha=random.randint(alpha//2, alpha), sigma=20) # apply augmentation
            cv2.imwrite(os.path.join(aug_images_dir,  new_name), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)) # save new image
            cv2.imwrite(os.path.join(aug_masks_GC_dir,  new_name), cv2.cvtColor(aug_mask_GC, cv2.COLOR_RGB2BGR)) # save new GC mask
            cv2.imwrite(os.path.join(aug_masks_Starch_dir,  new_name), cv2.cvtColor(aug_mask_Starch, cv2.COLOR_RGB2BGR)) # save new Starch mask
    return None


def data_split(aug_images_dir, r_train=0.8):
    """split dataset into train and val with defined ratio"""
    random.seed(0); np.random.seed(0) # set seed 
    
    aug_file_names = sorted(os.listdir(aug_images_dir), key=str.casefold)
    aug_file_names = [name for name in aug_file_names if any(name.endswith(file_type) for file_type in image_types)]
    
    train_size = int(len(aug_file_names)*r_train) # training size
    validation_size = int(len(aug_file_names)*(1-r_train)) # validation size
    
    aug_file_names_shuffle = aug_file_names.copy() # make a copy of list to prevent changes in place
    random.shuffle(aug_file_names_shuffle) # random shuffle file names   
    train_names = aug_file_names_shuffle[:train_size] # file names for training
    val_names = aug_file_names_shuffle[train_size:train_size + validation_size] # file names for validation
    print(f'train size={train_size}, validation size={validation_size}')
    return train_names, val_names


def mask4YOLO(mask, draw=False):
    """create YOLO format mask data: class_index p1.x p1.y p2.x p2.y p3.x p3.y"""
    height, width = mask.shape[:2] # height and width
    points = contour(mask)[1] # (x1, y1), (x2, y2), ...
    annotation = [0] # class index 0
    for i in range(len(points)):
        for x, y in points[i]:
            annotation.append(x/width) # normalize x between 0 to 1
            annotation.append(y/height) # normalize y between 0 to 1
    if draw == True:
        x, y, w, h = cv2.boundingRect(contour(mask)[1]) # for boundary box
        plt.imshow(cv2.rectangle(mask.copy(),(x,y),(x+w,y+h),(255,255,255), thickness=2), cmap=plt.cm.gray)
        # to visualize the boundary box
        return None
    return annotation


def labels4YOLO(aug_masks_GC_dir, names, out_dir):
    """convert the mask of both guard cells to a YOLO txt file"""
    os.makedirs(out_dir, exist_ok=True) # new dir for YOLO txt files
    for file in tqdm(names, total=len(names)):
        mask_GC = imread_rgb(os.path.join(aug_masks_GC_dir, file))
        annotation_GC1 = mask4YOLO(color_select(mask_GC, mask_GC, GC_Colors[1].mask_rgb)) # GC1 annotation list
        annotation_GC2 = mask4YOLO(color_select(mask_GC, mask_GC, GC_Colors[2].mask_rgb)) # GC2 annotation list
        new_name = os.path.splitext(file)[0] + '.txt' # 'file.txt'
        new_path = os.path.join(out_dir, new_name)
        with open(new_path, 'w') as txt_file:
            for item in annotation_GC1:
                txt_file.write(str(item) + ' ')
            txt_file.write('\n')  # add a line break in between the lists
            for item in annotation_GC2:
                txt_file.write(str(item) + ' ')
    return None


def data4YOLO(aug_images_dir, aug_masks_GC_dir, data_dir, r_train):
    """create YOLO format datasets for training and validation"""
    images_YOLO_dir = os.path.join(data_dir, 'images') # YOLO images dir
    labels_YOLO_dir = os.path.join(data_dir, 'labels') # YOLO labels dir
    train_images_dir = os.path.join(images_YOLO_dir, 'train'); os.makedirs(train_images_dir, exist_ok=True) # train images dir
    val_images_dir = os.path.join(images_YOLO_dir, 'val'); os.makedirs(val_images_dir, exist_ok=True) # val images dir
    train_labels_dir = os.path.join(labels_YOLO_dir, 'train') # train labels dir
    val_labels_dir = os.path.join(labels_YOLO_dir, 'val') # val labels dir
    train_names, val_names = data_split(aug_images_dir, r_train=r_train)
    for name in tqdm(train_names, total=len(train_names)):
        source = os.path.join(aug_images_dir, name) # source train image
        destination = os.path.join(train_images_dir, name) # destination train image
        shutil.copy2(source, destination) # copy source and paste to destination
    for file in tqdm(val_names, total=len(val_names)):
        source = os.path.join(aug_images_dir, file) # source validation image
        destination = os.path.join(val_images_dir, file) # destination validation image
        shutil.copy2(source, destination) # copy source and paste to destination
    labels4YOLO(aug_masks_GC_dir, train_names, train_labels_dir) # mask train image labels
    labels4YOLO(aug_masks_GC_dir, val_names, val_labels_dir) # mask validation image labels
    import yaml # to create ymal YOLO data info file
    data = {
        'path': os.path.abspath(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'GuardCell'
        },
        'nc': 1
    }
    with open(os.path.join(data_dir, 'YOLO data.yaml'), 'w') as file:
        documents = yaml.dump(data, file)
    return None


def json4COCO(aug_masks_GC_dir, names):
    """return the jason file for COCO format"""
    obj_count, images, annotations = 0, [], []
    for idx, name in tqdm(enumerate(names), total=len(names)):
        mask_GC = imread_rgb(os.path.join(aug_masks_GC_dir, name)) # load the mask
        height, width = mask_GC.shape[:2] # height and width
        images.append(dict(id=idx, file_name=name, height=height, width=width))

        for i in [1, 2]:
            points_GC = contour(color_select(mask_GC, mask_GC, GC_Colors[i].mask_rgb))[1] # (x1, y1), (x2, y2), ...
            px, py = [point[0][0] for point in points_GC], [point[0][1] for point in points_GC] #[x1, x2, ...], [y1, y2, ...]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)] # polygon: adding 0.5 shifting the x,y from the pixel corners to the pixel centers.
            poly = [p for x in poly for p in x] # flatten a nested list structure and store the flattened elements in the poly list
            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py)) # get bounding box
            data_anno = dict(image_id=idx, id=obj_count, category_id=0,
                        bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                        area=(x_max - x_min) * (y_max - y_min),
                        segmentation=[poly], iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(images=images, annotations=annotations, categories=[{'id':0, 'name': 'Guard Cell'}])
    return coco_format_json


def data4COCO(aug_images_dir, aug_masks_GC_dir, data_dir, r_train):
    """create COCO format datasets for training and validation"""
    train_dir = os.path.join(data_dir, 'train'); os.makedirs(train_dir, exist_ok=True) # COCO train dir
    val_dir = os.path.join(data_dir, 'val'); os.makedirs(val_dir, exist_ok=True) # COCO val dir
    train_names, val_names = data_split(aug_images_dir, r_train=r_train)
    for name in tqdm(train_names, total=len(train_names)):
        source = os.path.join(aug_images_dir, name) # source train image
        shutil.copy2(source, train_dir) # copy source and paste to destination
    for name in tqdm(val_names, total=len(val_names)):
        source = os.path.join(aug_images_dir, name) # source validation image
        shutil.copy2(source, val_dir) # copy source and paste to destination
    import json
    class NpEncoder(json.JSONEncoder):
        """check json encoding format"""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    json.dump(json4COCO(aug_masks_GC_dir, train_names), open(os.path.join(train_dir, 'COCO.json'), "w"), cls=NpEncoder)
    json.dump(json4COCO(aug_masks_GC_dir, val_names), open(os.path.join(val_dir, 'COCO.json'), "w"), cls=NpEncoder)
    return None


def creator4COCOYOLO(images_dir, masks_dir, data_dir, r_train, data_format='COCO'):
    """create instance segmentation dataset in COCO or YOLO format"""
    if data_format == 'COCO':
        data4COCO(images_dir, masks_dir, data_dir, r_train) # create COCO format dataset
    elif data_format == 'YOLO':
        data4YOLO(images_dir, masks_dir, data_dir, r_train) # create YOLO format dataset
    return None


def data4seg(root):
    """"create semantic segmentation dataset for both stomata types"""
    Base_dir = os.path.join(root, 'Data_MMseg')
    images_dir = os.path.join(Base_dir, 'images'); os.makedirs(images_dir, exist_ok=True) # images dir
    labels_dir = os.path.join(Base_dir, 'labels'); os.makedirs(labels_dir, exist_ok=True) # masks dir
    splits_dir = os.path.join(Base_dir, 'splits'); os.makedirs(splits_dir, exist_ok=True) # train/val file dir
    
    for folder_name in tqdm(['Aug_Images', 'Aug_masks_Starch'], total=2):
        source = os.path.join(root, folder_name) # copy source
        if folder_name== 'Aug_Images':
            destination = images_dir # destination
        elif folder_name== 'Aug_masks_Starch':
            destination = labels_dir # destination
        file_names = [name for name in os.listdir(source) if any(name.endswith(file_type) for file_type in image_types)]
        for name in file_names:
            shutil.copy2(os.path.join(source, name), os.path.join(destination, name)) # copy files to the common folder

    def annotation(mask):
        """convert the starch mask to gray scale and pixel of 1 or 0"""   
        mask = np.all(mask == Starch_Colors[1].mask_rgb, axis=-1) # where starch located
        black_image = np.zeros((mask.shape[:2]), dtype=np.uint8) # empty black images in (width, height)
        black_image[mask] = 1 # assign value 1 to where starch located, whilte other places are 0
        return black_image

    file_names = [name for name in os.listdir(labels_dir) if any(name.endswith(file_type) for file_type in image_types)]
    for name in tqdm(file_names, total=len(file_names)):
        mask = imread_rgb(os.path.join(labels_dir, name)) # load rgb mask
        gray = Image.fromarray(annotation(mask), mode='L') # rgb to one-hot encoding
        gray.putpalette(np.array([Starch_Colors[1].mask_rgb], dtype=np.uint8)) # for visualize mask
        gray.save(os.path.join(labels_dir, name)) # replace the rgb mask with one-hot encoded mask
    
    train_names, val_names = data_split(images_dir, r_train=0.8) # split train and validation
    with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
        f.writelines(os.path.splitext(name)[0] + '\n' for name in train_names) # creat a txt file for train
    with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:  
        f.writelines(os.path.splitext(name)[0] + '\n' for name in val_names) # create a txt file for validation
    return None


