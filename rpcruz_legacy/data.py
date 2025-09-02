import xml.etree.ElementTree as ET
from skimage.io import imread  # scikit-image
import numpy as np
import json
import os

class PascalVOC:
    num_classes = 20
    objects = [
        'person',  # person
        'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',  # animal
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',  # vehicle
        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor',  # indoor
    ]

    def __init__(self, root, fold, transform=None):
        fold = 'train' if fold == 'train' else 'val'
        self.root = os.path.join(root, 'toys', 'VOCdevkit', 'VOC2012')
        self.files = [f.rstrip() for f in open(os.path.join(self.root, 'ImageSets', 'Main', fold + '.txt'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        image = imread(os.path.join(self.root, 'JPEGImages', fname + '.jpg'))
        xml = ET.parse(os.path.join(self.root, 'Annotations', fname + '.xml'))
        labels = [self.objects.index(obj.find('./name').text) for obj in xml.findall('.//object')]
        bboxes = [(
            int(bbox.find('./xmin').text), int(bbox.find('./ymin').text), int(bbox.find('./xmax').text),
            int(bbox.find('./ymax').text)) for bbox in xml.findall('.//object/bndbox')]
        d = {'image': image, 'labels': labels, 'bboxes': bboxes}
        if self.transform:
            d = self.transform(**d)
        return d

class COCO:
    num_classes = 80
    objects = [
        1,  # person
        2, 3, 4, 5, 6, 7, 8, 9,  # vehicle
        10, 11, 13, 14, 15,  # outdoor
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  # animal
        27, 28, 31, 32, 33,  # accessory
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43,  # sports
        44, 46, 47, 48, 49, 50, 51,  # kitchen
        52, 53, 54, 55, 56, 57, 58, 59, 60, 61,  # food
        62, 63, 64, 65, 67, 70,  # furniture
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,  # electronic
        84, 85, 86, 87, 88, 89, 90,  # indoor
    ]

    def __init__(self, root, fold, transform=None):
        self.fold = 'train' if fold == 'train' else 'val'
        self.root = os.path.join(root, 'toys', 'coco')
        ann = json.load(open(os.path.join(self.root, 'annotations', f'instances_{self.fold}2017.json')))
        self.ix = [img['id'] for img in ann['images']]
        self.files = {img['id']: img['file_name'] for img in ann['images']}
        self.labels = {}
        self.bboxes = {}
        for a in ann['annotations']:
            # some COCO objects are removed because they are badly represented
            if a['category_id'] not in self.objects: continue
            # remove bboxes too small, otherwise we get numerical reasons because xmin=xmax
            if a['bbox'][2] < 10 or a['bbox'][3] < 10: continue
            i = a['image_id']
            self.labels[i] = self.labels.get(i, []) + [self.objects.index(a['category_id'])]
            self.bboxes[i] = self.bboxes.get(i, []) + [(  # xywh -> xyxy
                a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3])]
        self.transform = transform

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, i):
        i = self.ix[i]
        fname = self.files[i]
        image = imread(os.path.join(self.root, f'{self.fold}2017', fname))
        if len(image.shape) == 2:
            image = np.repeat(image[..., None], 3, -1)
        d = {'image': image, 'labels': self.labels.get(i, []), 'bboxes': self.bboxes.get(i, [])}
        if self.transform:
            d = self.transform(**d)
        return d

class KITTI:
    num_classes = 3
    class_map = {'Car': 0, 'Van': 0, 'Cyclist': 1, 'Pedestrian': 2, 'Person_sitting': 2}

    def __init__(self, root, fold, transform=None):
        self.root = os.path.join(root, 'auto', 'kitti', 'object', 'training')
        self.files = sorted(os.listdir(os.path.join(self.root, 'image_2')))
        rng = np.random.default_rng(123)
        ix = rng.choice(len(self.files), len(self.files), False)
        ix = ix[:int(len(ix)*0.8)] if fold == 'train' else ix[int(len(ix)*0.8):]
        self.files = [self.files[i] for i in ix]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_fname = os.path.join(self.root, 'image_2', self.files[i])
        gt_fname = os.path.join(self.root, 'label_2', self.files[i][:-3] + 'txt')
        image = imread(img_fname)
        labels = [line.split()[0] for line in open(gt_fname)]
        keeps = [label in self.class_map for label in labels]
        labels = [self.class_map[label] for keep, label in zip(keeps, labels) if keep]
        bboxes = [[float(v) for v in line.split()[4:8]] for keep, line in zip(keeps, open(gt_fname)) if keep]
        d = {'image': image, 'labels': labels, 'bboxes': bboxes}
        if self.transform:
            d = self.transform(**d)
        return d
