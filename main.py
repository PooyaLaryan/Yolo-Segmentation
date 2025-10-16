from ultralytics import YOLO
import cv2
import torch

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0)) 


model = YOLO("yolo11n-seg.pt")

video_path = "v1.ts" 
output_video_path = "output.mp4"

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # کدک MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not cap.isOpened():
    print("❌ Cannot open .ts video file")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if success:
        result = model(frame, device=0, classes=[0])
        annotated_frame = result[0].plot(labels=False, boxes=False)
        cv2.imshow("TS Video", annotated_frame)
        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()