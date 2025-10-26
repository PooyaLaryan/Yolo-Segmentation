import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

input_path = "val_img/3.jpg"
output_path = "3-mask.jpg"

image = cv2.imread(input_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_tensor = F.to_tensor(image_rgb).to(device)

with torch.no_grad():
    outputs = model([img_tensor])[0]

CONF_THRESH = 0.7
masks = outputs["masks"]
labels = outputs["labels"]
scores = outputs["scores"]

mask_overlay = np.zeros_like(image, dtype=np.uint8)

for mask, label, score in zip(masks, labels, scores):
    if label == 1 and score > CONF_THRESH:  # 'person' class = 1
        m = mask[0].mul(255).byte().cpu().numpy()
        mask_bin = m > 128
        mask_overlay[mask_bin] = (0, 255, 0)  # green overlay

result = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)

cv2.imwrite(output_path, result)
cv2.imshow("Masked People", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Saved segmented image to: {output_path}")
