import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
import torchvision.transforms as T
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_instance_segmentation(num_classes):
    # Load a pre-trained model for instance segmentation
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def get_transforms():
    return T.Compose([
        T.ToTensor(),  # Converts PIL image or numpy.ndarray to tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
        ])


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

if __name__ == '__main__':
    
    # Define your dataset and dataloaders
    dataset = CustomCocoDataset(
            img_folder='./dataset/Images/batch1',
            ann_file='./dataset/Annotations/batch1.json',
            transforms=get_transforms()
            )

    dataset2 = CustomCocoDataset(
            img_folder='./dataset/Images/batch2',
            ann_file='./dataset/Annotations/batch2.json',
            transforms=get_transforms()
            )

    dataset3 = CustomCocoDataset(
            img_folder='./dataset/Images/batch3',
            ann_file='./dataset/Annotations/batch3.json',
            transforms=get_transforms()
            )

    dataset4 = CustomCocoDataset(
            img_folder='./dataset/Images/batch4',
            ann_file='./dataset/Annotations/batch4.json',
            transforms=get_transforms()
            )

    # Combine all datasets into one
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([dataset, dataset2, dataset3, dataset4])

    # Dataloader with num_workers > 0 requires __main__ guard on Windows
    def collate_fn(batch):
        return tuple(zip(*batch))

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # Define the model
    model = get_model_instance_segmentation(num_classes=2)

    # Move model to device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Define optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 10
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_maskrcnn_model.pth')
            print(f"Saved best model with loss: {best_loss:.4f}")

        lr_scheduler.step()

