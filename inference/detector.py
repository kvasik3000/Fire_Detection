import io

import av
import cv2
import numpy as np
import onnxruntime


class Detector:

    def __init__(
        self,
        model_path: str,
        size: int | tuple[int, int] = 640,
        classes: dict[int, str] = {0: "fire", 1: "smoke"},
    ):
        """Initialize the MyDetector object.

        Args:
            model_path (str): Path to the model file.
            size (int | tuple): Size of the input image.
            classes (dict): Class IDs and their names

        Returns:
            None
        """
        self._session = onnxruntime.InferenceSession(model_path)
        self.input_name = self._session.get_inputs()[0].name
        self.output_names = [self._session.get_outputs()[0].name]
        self.classes = classes
        self._size = size if isinstance(size, tuple) else (size, size)

    def forward(self, rgb_img: np.ndarray):
        """Perform forward pass on the input image.

        Args:
            rgb_img (ndarray): Input RGB image.

        Returns:
            Output of the forward pass.
        """
        input_image = cv2.resize(rgb_img, self._size)
        input_image = np.transpose(input_image, [2, 0, 1])
        input_image = input_image.astype("float32") / 255
        input_image = np.expand_dims(input_image, axis=0)

        return self._session.run(self.output_names, {self.input_name: input_image})[0]

    def post_process(
        self,
        output: tuple,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        score_threshold: float = 0.5,
    ):
        """Post-process the output of the model.

        Args:
            output (tuple): Output of the model.

        Returns:
            tuple: Processed bounding boxes and class IDs.
        """
        boxes = []
        scores = []
        class_ids = []

        for pred in output:
            pred = np.transpose(pred)

            for box in pred:
                x, y, w, h = box[:4]
                x1 = x - w / 2
                y1 = y - h / 2
                boxes.append([x1, y1, w, h])
                idx = np.argmax(box[4:])
                scores.append(box[idx + 4])
                class_ids.append(idx)

        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, iou_threshold)
        indices = indices[[scores[i] > score_threshold for i in indices]] if len(indices) > 0 else []

        return (
            np.array(boxes)[indices],
            np.array(class_ids)[indices],
            np.array(scores)[indices],
        )

    def draw_detections(self, img: np.ndarray, box: np.ndarray, score: float, class_id: int):
        """Draws bounding boxes and labels on the input image based on the detected
        objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Create the label text with class name and score
        keep = True

        try:
            label = f"{self.classes[class_id]}: {score:.2f}"
        except KeyError:
            keep = False

        if keep:
            # Retrieve the color for the class ID
            color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
            color = color_palette[class_id]

            # Draw the bounding box on the image
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(
                img,
                (int(label_x), int(label_y - label_height)),
                (int(label_x + label_width), int(label_y + label_height)),
                color,
                cv2.FILLED,
            )

            # Draw the label text on the image
            cv2.putText(
                img,
                label,
                (int(label_x), int(label_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def draw_box(self, img: np.ndarray):
        """Draw bounding boxes on the input image.

        Args:
            img (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with bounding boxes drawn.
        """
        bboxes, class_ids, scores = self.post_process(self.forward(img))
        input_image = img.copy()

        inp_height, inp_width, _ = input_image.shape
        img_width, img_height = self._size

        for i in range(len(bboxes)):
            # Get the box, score, and class ID corresponding to the index
            box = bboxes[i]
            gainW = inp_width / img_width
            gainH = inp_height / img_height
            pad = (
                round((img_width - inp_width / gainW) / 2),
                round((img_height - inp_height / gainH) / 2),
            )
            box[0] = (box[0] - pad[0]) * gainW
            box[1] = (box[1] - pad[1]) * gainH
            box[2] = box[2] * gainW
            box[3] = box[3] * gainH

            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        return input_image

    def video_run(self, cap: cv2.VideoCapture):
        """Process video frames and draw bounding boxes.

        Args:
            cap: Video capture object.

        Returns:
            io.BytesIO: In-memory file containing the processed video.
        """

        output_memory_file = io.BytesIO()
        output_f = av.open(output_memory_file, "w", format="mp4")  # Open "in memory file" as MP4 video output
        stream = output_f.add_stream("h264", 60)  # Add H.264 video stream to the MP4 container, with framerate = fps.

        ret, frame = cap.read()

        while ret:
            res_img = self.draw_box(frame)
            res_img = av.VideoFrame.from_ndarray(res_img, format="bgr24")  # Convert image from NumPy Array to frame.
            packet = stream.encode(res_img)  # Encode video frame
            output_f.mux(packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).
            ret, frame = cap.read()

        # Flush the encoder
        packet = stream.encode(None)
        output_f.mux(packet)
        output_f.close()

        return output_memory_file

    def __call__(self, rgb_img):
        """Call the MyDetector object to process an input image.

        Args:
            rgb_img (numpy.ndarray): Input RGB image.

        Returns:
            tuple: Processed bounding boxes and class IDs.
        """
        return self.draw_box(rgb_img)
