import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

input_path = "videos/4.mp4"
output_path = "output_masked_4.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
CONF_THRESH = 0.7

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_tensor = F.to_tensor(frame).to(device)
    with torch.no_grad():
        outputs = model([img_tensor])[0]

    masks = outputs['masks']
    labels = outputs['labels']
    scores = outputs['scores']

    mask_overlay = np.zeros_like(frame, dtype=np.uint8)

    for mask, label, score in zip(masks, labels, scores):
        if label == 1 and score > CONF_THRESH:  # 1 = person class in COCO
            m = mask[0].mul(255).byte().cpu().numpy()
            mask_bin = m > 128
            mask_overlay[mask_bin] = (0, 255, 0)

    result = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)

    out.write(result)
    cv2.imshow("Masked People", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved masked video to: {output_path}")
