# (C) 2025 Ricardo Cruz <rpcruz@fe.up.pt>

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('output')
parser.add_argument('--mixmatch', action='store_true')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--sup-batchsize', type=int, default=8)
parser.add_argument('--unsup-batchsize', type=int, default=32)
parser.add_argument('--imgsize', type=int, default=640)
args = parser.parse_args()

import torch, torchvision
import ultralytics
import data, myyolo
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchmetrics
from time import time
import itertools
from types import SimpleNamespace

############################ DATA ############################

bbox_params = A.BboxParams('pascal_voc', label_fields=['labels'])
preprocess = A.Compose([
    A.Resize(args.imgsize, args.imgsize),
])
augment = A.ReplayCompose([
    # TODO: add support for geometric transformations
    #A.Rotate(30),
    #A.HorizontalFlip(),
    A.RandomBrightnessContrast(p=1),
])
postprocess = A.Compose([
    A.Normalize(0, 1),
    ToTensorV2(),
])

dataset = getattr(data, args.dataset)('/data', preprocess)
generator = torch.Generator().manual_seed(123)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
sup_train_dataset, unsup_train_dataset = torch.utils.data.random_split(train_dataset, [0.1, 0.9])

sup_train_dataloader = torch.utils.data.DataLoader(sup_train_dataset, args.sup_batchsize, True, num_workers=4, pin_memory=True, collate_fn=lambda x: x)
unsup_train_dataloader = torch.utils.data.DataLoader(unsup_train_dataset, args.unsup_batchsize, True, num_workers=4, pin_memory=True, collate_fn=lambda x: x)
test_dataloader = torch.utils.data.DataLoader(test_dataset, args.sup_batchsize, num_workers=4, pin_memory=True, collate_fn=lambda x: x)

############################ LOOP ############################

# YOLO models, from smaller to bigger:
# yolo11n yolo11s yolo11m yolo11l yolo11x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = myyolo.MyYOLO('yolo11n', nc=dataset.num_classes, input_resolution=args.imgsize)
model.to(device)
opt = torch.optim.AdamW(model.parameters())

# v8DetectionLoss is also used for YOLO11
model.yolo.model.args = SimpleNamespace(box=0.05, cls=0.5, dfl=1.5)
criterion = ultralytics.utils.loss.v8DetectionLoss(model.yolo.model)

for epoch in range(args.epochs):
    # train
    tic = time()
    avg_sup_loss = avg_unsup_loss = 0
    model.set_trainable()
    semisup_n_iter = max(len(sup_train_dataloader), len(unsup_train_dataloader))
    sup_dataloader_iter = itertools.cycle(sup_train_dataloader) if len(sup_train_dataloader) < len(unsup_train_dataloader) else iter(sup_train_dataloader)
    unsup_dataloader_iter = itertools.cycle(unsup_train_dataloader) if len(unsup_train_dataloader) < len(sup_train_dataloader) else iter(unsup_train_dataloader)

    for sup_batch, unsup_batch in zip(sup_dataloader_iter, unsup_dataloader_iter):
        sup_batch = [postprocess(**augment(**d)) for d in sup_batch]
        sup_imgs = torch.stack([d['image'].to(device) for d in sup_batch])
        # predictions are in the format: [(B, C, H, W)]
        # len(list) = 3 (multi-scale)
        # B=batch size, H/W=depend on the scale, C=num_classes+16*4
        # 16*4 is because the model instead of taking bounding boxes as a regresion problem
        # it discretizes each coordinate into 16 probabilities.
        sup_preds = model(sup_imgs)

        # the loss assumes that the batch is formatted as:
        # {batch_idx: image ids, cls: class number, bboxes: xcycwh normalized 0-1}
        # it is a quite clever way to handle the fact that each image has varying number of bounding boxes
        # - they are then able to concatenate this into a tensor
        gt = {
            'batch_idx': torch.tensor([i for i, d in enumerate(sup_batch) for _ in d['labels']]),
            'cls': torch.tensor([l for d in sup_batch for l in d['labels']]),
            'bboxes': torch.tensor([b for d in sup_batch for b in d['bboxes']]),
        }
        gt['bboxes'] = torchvision.ops.box_convert(gt['bboxes'], 'xyxy', 'cxcywh') / args.imgsize
        sup_losses, _  = criterion(sup_preds, gt)
        sup_loss = sup_losses.sum()
        avg_sup_loss += float(sup_loss) / len(sup_train_dataloader)

        # TODO: would be interesting to also implement FixMatch, which has already been adapted to object detection by
        # these two papers:
        # [1] https://arxiv.org/abs/2210.09919
        # [2] https://arxiv.org/abs/2208.00400
        # [3] https://arxiv.org/abs/2204.07300
        # the following is some code I implemented for FCOS (not YOLO!) based on [3] - please note that maybe it's easier to ignore
        # this code and start again :D
        '''
        if args.fixmatch:  # FIXME: port to YOLO
            # the Adaptive Filter part of the DenSe paper [3]
            model.eval()
            unsup = [weak_augment(image=d['image'], labels=[], bboxes=[]) for d in unsup]
            unsup_imgs = torch.stack([torch.tensor(d['image'], device=device).permute(2, 0, 1)/255 for d in unsup])
            with torch.no_grad():
                # should we disable score_threshold and nms_threshold?
                pseudo_targets = model(unsup_imgs)
                cls_masks = model.head.cls_masks

            # filter cases of degenerate bboxes (x1 == x2) because those give an error on FCOS
            pseudo_targets = [{
                'labels': d['labels'][(d['boxes'][:, 0] < d['boxes'][:, 2]) & (d['boxes'][:, 1] < d['boxes'][:, 3])],
                'bboxes': d['boxes'][(d['boxes'][:, 0] < d['boxes'][:, 2]) & (d['boxes'][:, 1] < d['boxes'][:, 3])]} for d in pseudo_targets]
            model.train()
            cls_masks = [(mask <= 0.1) | (mask >= 0.9) for mask in cls_masks]
            # strong augment on top of the weak images, label/bboxes predictions for those images, and the threshold score masks
            strong_unsup = [strong_augment(image=d['image'], labels=pseudo['labels'].cpu().numpy(), bboxes=pseudo['bboxes'].cpu().numpy(), mask=mask.cpu().numpy()) for d, pseudo, mask in zip(unsup, pseudo_targets, cls_masks)]
            model.head.weight_matrices = [torch.tensor(d['mask']) for d in strong_unsup]
            strong_unsup_imgs = torch.stack([torch.tensor(d['image'], device=device).permute(2, 0, 1)/255 for d in strong_unsup])
            weak_unsup_targets = [{'labels': torch.tensor(d['labels'], dtype=int, device=device), 'boxes': torch.tensor(d['bboxes'], device=device)} for d in strong_unsup]
            unsup_losses = model(strong_unsup_imgs, weak_unsup_targets)
            unsup_loss = sum(unsup_losses.values())
            model.head.weight_matrices = None
        '''
        
        if args.mixmatch:
            # https://proceedings.neurips.cc/paper_files/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html
            # we are using K=8 (instead of K=2 like the paper) to make it more comparable with ReMixMatch
            K = 8
            T = 0.5
            alpha = 0.75
            pseudoimgs = [None]*K
            pseudolabels_perscale = [[None]*K for _ in range(3)]
            # TODO: add support for geometric transformations
            #       -> maybe easier to apply the same transformation for the entire batch
            #       -> then invert the predictions
            for i in range(K):
                each_unsup_batch = [postprocess(**augment(**d)) for d in unsup_batch]
                each_unsup_imgs = torch.stack([d['image'].to(device) for d in each_unsup_batch])
                # remember, preds=[(B, C, H, W)]*3
                # where C=[Pclasses, Pbboxes]
                with torch.no_grad():
                    preds = model(each_unsup_imgs)
                pseudoimgs[i] = each_unsup_imgs
                for j in range(3):
                    pseudolabels_perscale[j][i] = preds[j]
            # pseudo-labels: average
            pseudolabels_perscale = [sum(p)/K for p in pseudolabels_perscale]
            # pseudo-labels: sharpen
            pseudolabels_perscale = [(p**(1/T)) / torch.sum(p**(1/T), 1, True) for p in pseudolabels_perscale]
            # concat
            pseudoimgs = torch.cat(pseudoimgs)
            pseudolabels_perscale = [p.repeat(K, 1, 1, 1) for p in pseudolabels_perscale]
            # mixup
            # TODO: I only mixup between unsup; misses mixup between sup and unsup
            # TODO: Beta distribution instead of Uniform
            lamda = torch.rand(len(pseudoimgs), device=device)[:, None, None, None]
            ix = torch.randperm(len(pseudoimgs), device=device)
            mixed_imgs = lamda*pseudoimgs + (1-lamda)*pseudoimgs[ix]
            mixed_pseudolabels = [lamda*p + (1-lamda)*p[ix] for p in pseudolabels_perscale]
            # loss
            preds = model(mixed_imgs)
            unsup_loss = sum(torch.mean((l1-l2)**2) for l1, l2 in zip(preds, pseudolabels_perscale))/3
            avg_unsup_loss += float(unsup_loss) / len(unsup_train_dataloader)
        else:
            unsup_loss = 0

        loss = sup_loss + unsup_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Sup loss: {avg_sup_loss} - Unsup loss: {avg_unsup_loss}')
    # evaluate
    tic = time()
    map = torchmetrics.detection.mean_ap.MeanAveragePrecision()
    model.set_eval()
    for sup_batch in test_dataloader:
        sup_batch = [postprocess(**d) for d in sup_batch]
        imgs = torch.stack([d['image'].to(device) for d in sup_batch])
        targets = [{'labels': torch.tensor(d['labels'], dtype=int, device=device), 'boxes': torch.tensor(d['bboxes'], device=device)} for d in sup_batch]
        with torch.no_grad():
            preds = model(imgs, True)
        map.update(preds, targets)

    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Test mAP: {map.compute()['map'].item()}')

print(f'Saving model...')
torch.save(model.cpu(), args.output)
