from torchvision import transforms

from .kitti import KITTIDepthDataset, align_depth_target
from .tartanair import TartanAirDataset, align_depth_target

DATASET_CLASS = {
    'kitti': KITTIDepthDataset,
    'tartanair': TartanAirDataset,
}


def get_dataset_transform(dataset, input_size):
    common = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if dataset == 'tartanair':
        spatial = [transforms.Resize((input_size, input_size))]
    else:
        spatial = [transforms.CenterCrop((input_size, input_size))]
    return transforms.Compose(spatial + common)


def make_dataset(dataset, root_dir, mode='train', input_size=224):
    cls = DATASET_CLASS[dataset]
    return cls(root_dir=root_dir,
               transform=get_dataset_transform(dataset, input_size),
               mode=mode)