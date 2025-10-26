import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import threading
import queue
from time import sleep
import sys

# CONFIGURATION & CONSTANTS
INPUT_PATH = "videos/4.mp4"
OUTPUT_PATH = "output_batch_4.mp4"
INFERENCE_RES = (960, 540)
BATCH_SIZE = 4
CONF_THRESH = 0.7
USE_FP16 = True
USE_COMPILE = True
DISPLAY = False
mask_color = (255, 0, 0) # Red
MASK_ALPHA = 0.4

# MODEL INITIALIZATION
print("🔧 Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
model = model.to(device).eval()

if USE_FP16:
    model.half()
    print("✅ Model using FP16 precision.")

IS_WINDOWS = sys.platform == "win32"
if USE_COMPILE and not IS_WINDOWS:
    try:
        model = torch.compile(model)
        print("✅ Model compiled for optimized performance (Linux/macOS).")
    except Exception as e:
        print(f"⚠️ torch.compile not available or failed: {e}")
elif USE_COMPILE and IS_WINDOWS:
    print("⚠️ Skipping torch.compile on Windows as it is not supported.")

# VIDEO I/O SETUP
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {INPUT_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎬 Input video: {original_width}x{original_height} @ {fps:.2f} FPS ({total_frames} frames)")
print(f"🧠 Inference resolution: {INFERENCE_RES[0]}x{INFERENCE_RES[1]}")

# The output video writer is initialized with the ORIGINAL resolution.
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (original_width, original_height))


# full-resolution frames are read from the video file.
frame_queue = queue.Queue(maxsize=BATCH_SIZE * 2)
stop_flag = False

def reader_thread():
    """Reads frames from the video file and puts them into a queue."""
    global stop_flag
    while not stop_flag:
        if frame_queue.full():
            sleep(0.01)  # Wait if the queue is full to avoid high memory usage
            continue

        ret, frame = cap.read()
        if not ret:
            print("\nReader thread: Reached end of video.")
            break
        # Put the ORIGINAL, full-resolution frame into the queue.
        frame_queue.put(frame)

    cap.release()
    frame_queue.put(None) # Sentinel value to signal the end

threading.Thread(target=reader_thread, daemon=True).start()

print("🚀 Starting GPU inference...")
frame_count = 0
with torch.no_grad():
    while not stop_flag:
        batch_tensors = []
        original_frames_batch = []

        # 1. Batch Preparation: Collect frames and resize for inference
        for _ in range(BATCH_SIZE):
            original_frame = frame_queue.get()
            if original_frame is None:  # End of stream
                stop_flag = True
                break

            original_frames_batch.append(original_frame)

            # Resize for the model
            resized_frame = cv2.resize(original_frame, INFERENCE_RES, interpolation=cv2.INTER_AREA)
            tensor = F.to_tensor(resized_frame)
            if USE_FP16:
                tensor = tensor.half()
            tensor = tensor.to(device, non_blocking=True)
            batch_tensors.append(tensor)

        if not batch_tensors:
            break

        # 2. Model Inference
        outputs = model(batch_tensors)

        # 3. Post-processing and Compositing
        for i, output in enumerate(outputs):
            original_frame = original_frames_batch[i]

            # Create a blank canvas at the INFERENCE resolution to draw masks on.
            mask_canvas = np.zeros((INFERENCE_RES[1], INFERENCE_RES[0], 3), dtype=np.uint8)


            masks, labels, scores = output["masks"], output["labels"], output["scores"]
            for mask, label, score in zip(masks, labels, scores):
                # Filter for 'person' class (label=1) with high confidence
                if label == 1 and score > CONF_THRESH:
                    m = mask[0].mul(255).byte().cpu().numpy()
                    mask_bin = m > 128
                    mask_canvas[mask_bin] = mask_color

            # Upscale the mask canvas back to the ORIGINAL video resolution.
            upscaled_mask = cv2.resize(mask_canvas, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

            # Blend the upscaled mask with the original, high-quality frame.
            result_frame = cv2.addWeighted(original_frame, 1.0, upscaled_mask, MASK_ALPHA, 0)
            out.write(result_frame)

            if DISPLAY:
                display_frame = cv2.resize(result_frame, (1280, 720))
                cv2.imshow("Masked People (Preview)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_flag = True
                    break

            frame_count += 1

        print(f"\rProcessed {frame_count}/{total_frames} frames...", end="")

stop_flag = True
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Saved high-quality masked video to: {OUTPUT_PATH}")
