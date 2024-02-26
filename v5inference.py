from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('weights/v5200.pt')
# model.to("cuda")

VIDEO_PATH = "fire001_speedup_zipped.mp4"
video_output = "fire001_inf_v5.mp4"

video_stream_in = cv2.VideoCapture(VIDEO_PATH)

video_width = int(video_stream_in.get(3))
video_height = int(video_stream_in.get(4))
framerate = int(video_stream_in.get(5))
total_frames = int(video_stream_in.get(7))


video_stream_out = cv2.VideoWriter(
        filename=video_output,
        fourcc=0x7634706D,
        fps=framerate,
        frameSize=(video_width, video_height),
    )


while video_stream_in.isOpened():
    success, frame = video_stream_in.read()
    print(success)
    if success:
        results = model(frame, conf=0.3)
        annotated_frame = results[0].plot()
        video_stream_out.write(annotated_frame)
    else:
        break

video_stream_in.release()
video_stream_out.release()



# import torch
# import cv2

# # Load a model
# model = torch.hub.load("ultralytics/yolov5", 'custom', path="F:/Programming/Fire-detection/weights/v5100.pt", force_reload=True)
# # model.load_state_dict(torch.load("F:/Programming/Fire-detection/weights/v5100.pt"))
# model.to("cuda")

# VIDEO_PATH = "fire001_speedup_zipped.mp4"
# video_output = "fire001_inf_v5.mp4"

# video_stream_in = cv2.VideoCapture(VIDEO_PATH)

# video_width = int(video_stream_in.get(3))
# video_height = int(video_stream_in.get(4))
# framerate = int(video_stream_in.get(5))
# total_frames = int(video_stream_in.get(7))


# video_stream_out = cv2.VideoWriter(
#         filename=video_output,
#         fourcc=0x7634706D,
#         fps=framerate,
#         frameSize=(video_width, video_height),
#     )


# while video_stream_in.isOpened():
#     success, frame = video_stream_in.read()
#     print(success)
#     if success:
#         results = model(frame)
#         results.render()
#         annotated_frame = results.ims[0]
#         video_stream_out.write(annotated_frame)
#     else:
#         break

# video_stream_in.release()
# video_stream_out.release()
