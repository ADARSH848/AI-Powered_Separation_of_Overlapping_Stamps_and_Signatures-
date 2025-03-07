import os


# Correct file paths
batch_dirs = [
    './dataset/images/batch1',
    './dataset/images/batch2',
    './dataset/images/batch3',
    './dataset/images/batch4'
]
annotation_files = [
    './dataset/annotations/batch1.json',
    './dataset/annotations/batch2.json',
    './dataset/annotations/batch3.json',
    './dataset/annotations/batch4.json'
]

for img_dir, ann_file in zip(batch_dirs, annotation_files):
    if not os.path.exists(img_dir):
        print(f"Image directory not found: {img_dir}")
    if not os.path.exists(ann_file):
        print(f"Annotation file not found: {ann_file}")
