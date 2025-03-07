import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CustomCocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        
        # Image transformation
        if self.transforms:
            img = self.transforms(img)
        
        # Adjust target format
        image_id = target[0]['image_id']
        boxes = [obj['bbox'] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([obj['category_id'] for obj in target], dtype=torch.int64)
        masks = [self.coco.annToMask(obj) for obj in target]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # Prepare target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id]),
            'area': torch.tensor([obj['area'] for obj in target], dtype=torch.float32),
            'iscrowd': torch.tensor([obj['iscrowd'] for obj in target], dtype=torch.int64)
        }
        
        return img, target
transform = T.Compose([
    T.ToTensor(),
])

batch_dirs = ['./dataset/Images/batch1', './dataset/Images/batch2', './dataset/Images/batch3', './dataset/Images/batch4', './dataset/Images/batch5', './dataset/Images/batch6']
annotation_files = ['C:/Users/dasar/Desktop/HackaThon/python/dataset/Annotations/batch2.json','C:/Users/dasar/Desktop/HackaThon/python/dataset/Annotations/batch3.json','C:/Users/dasar/Desktop/HackaThon/python/dataset/Annotations/batch4.json','C:/Users/dasar/Desktop/HackaThon/python/dataset/Annotations/batch5.json','C:/Users/dasar/Desktop/HackaThon/python/dataset/Annotations/batch6.json']

datasets = []
for img_dir, ann_file in zip(batch_dirs, annotation_files):
    dataset = CustomCocoDataset(
        img_folder=os.path.join('dataset/images', img_dir),
        ann_file=os.path.join('dataset/annotations', ann_file),
        transforms=transform
    )
    datasets.append(dataset)

from torch.utils.data import ConcatDataset
dataset = ConcatDataset(datasets)

# DataLoader
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)

model = maskrcnn_resnet50_fpn(pretrained=True)

# Modify the classifier and mask predictor for your dataset's classes (including background)
num_classes = 3  # Background + Stamp + Sign
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Variables to track the best model
best_loss = float('inf')
best_model_path = 'C:/Users/dasar/Desktop/HackaThon/models'
num_epochs = 10

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    avg_epoch_loss = epoch_loss / len(data_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss}")

    # Save the best model
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with loss: {best_loss}")

print("Training complete!")
print(f"Best model saved as: {best_model_path}")



