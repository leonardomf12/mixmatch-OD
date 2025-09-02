import numpy as np

def invert_bboxes(replay, image_width, bboxes):
    for a in replay['transforms']:
        params = a['params']
        if params is None: continue
        if a['__class_fullname__'] in ['RandomBrightnessContrast', 'Sharpen', 'Solarize']:
            continue
        if a['__class_fullname__'] == 'Rotate':
            # bboxes -> points
            x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            corners = np.stack([
                np.stack([x_min, y_min], axis=1),
                np.stack([x_max, y_min], axis=1),
                np.stack([x_max, y_max], axis=1),
                np.stack([x_min, y_max], axis=1)
            ], axis=1)
            ones = np.ones((*corners.shape[:-1], 1))
            corners = np.concatenate([corners, ones], axis=-1)
            corners = corners.reshape(-1, 3).T
            # inverse matrix
            inv_matrix = np.linalg.inv(params['bbox_matrix'])
            corners = inv_matrix @ corners
            # normalize homogeneous coordinates
            corners /= corners[2, :]
            corners = corners[:2, :]
            corners = corners.T.reshape(-1, 4, 2)
            # points -> bboxes
            xmin = np.min(corners[..., 0], 1)
            ymin = np.min(corners[..., 1], 1)
            xmax = np.max(corners[..., 0], 1)
            ymax = np.max(corners[..., 1], 1)
            bboxes = np.stack((xmin, ymin, xmax, ymax), 1)
        elif a['__class_fullname__'] == 'HorizontalFlip':
            bboxes[:, 0] = image_width - bboxes[:, 2]
            bboxes[:, 2] = image_width - bboxes[:, 0]
        else:
            print('Warning: unknown transform:', a['__class_fullname__'])