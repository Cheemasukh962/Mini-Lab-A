"""
EEC174 Project A1 - Phase 2: YOLO Object Detection & MOT on Videos

Runs YOLOv3 (OpenCV DNN) on every frame of an input video, keeps only
'person' detections, tracks them with SORT, and writes an annotated video
showing each person's track ID and the total number of people counted.

Example:
    python yolo_counter.py -w yolo_files/yolov3.weights -c yolo_files/yolov3.cfg \
        -l yolo_files/coco.names -v mot_vid/MOTS20-09-raw.mp4
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np
from sort import *

# Detection parameters (same values as the YOLO tutorial)
probability_minimum = 0.5   # minimum confidence to keep a detection
threshold = 0.3             # IoU threshold used by non-maximum suppression


def parse_args():
    """Read and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description='Count people in a video using YOLOv3 + SORT multi-object tracking.')
    parser.add_argument('-w', '--weights', required=True, help='path to YOLO weights file (yolov3.weights)')
    parser.add_argument('-c', '--config', required=True, help='path to YOLO config file (yolov3.cfg)')
    parser.add_argument('-l', '--labels', required=True, help='path to class labels file (coco.names)')
    parser.add_argument('-v', '--video', required=True, help='path to input video (.mp4)')
    parser.add_argument('-o', '--output', default=os.path.join('mot_vid', 'MOTS20-09-result.mp4'),
                        help='path to save annotated video (default: mot_vid/MOTS20-09-result.mp4)')
    args = parser.parse_args()

    # Exit safely with usage info if any input file is missing or the wrong type
    for name in ('weights', 'config', 'labels', 'video'):
        path = getattr(args, name)
        if not os.path.isfile(path):
            parser.error('{} file not found: {}'.format(name, path))
    if not args.video.lower().endswith('.mp4'):
        parser.error('video must be an .mp4 file: {}'.format(args.video))
    return args


def load_network(config_path, weights_path):
    """Load YOLOv3 into OpenCV and get the names of the YOLO output layers."""
    network = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layers = network.getLayerNames()
    try:
        yolo_layers = [layers[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    except (IndexError, TypeError):
        # Newer OpenCV versions return a flat array (fix given in the README)
        yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']
    return network, yolo_layers


def load_labels(labels_path):
    """Read class names (one per line) from coco.names."""
    with open(labels_path) as f:
        return [line.strip() for line in f if line.strip()]


def detect_people(network, yolo_layers, image, person_class):
    """
    Run YOLO on one frame and return person detections after NMS.
    Returns a numpy array of rows [x1, y1, x2, y2, confidence] (the format SORT expects).
    """
    # Convert frame to blob and pass it through the network
    input_blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    network.setInput(input_blob)
    output = network.forward(yolo_layers)

    bounding_boxes = []
    confidences = []
    h, w = image.shape[:2]

    # Get bounding boxes and confidences, keeping only the person class
    for result in output:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if class_current == person_class and confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))

    # Non-maximum suppression removes duplicate boxes on the same person
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    # Convert [x, y, w, h] -> [x1, y1, x2, y2, score] for SORT
    detections = []
    if len(results) > 0:
        for i in np.array(results).flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            detections.append([x_min, y_min, x_min + box_width, y_min + box_height, confidences[i]])

    if len(detections) == 0:
        return np.empty((0, 5))  # SORT must still be called on frames with no detections
    return np.array(detections)


def draw_tracks(image, tracks, total_people, colour_box):
    """Draw each tracked box with its ID, plus the running people counter."""
    for x1, y1, x2, y2, track_id in tracks:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour_box, 2)
        cv2.putText(image, str(int(track_id)), (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour_box, 2)

    text_count = 'People: {}'.format(total_people)
    cv2.putText(image, text_count, (image.shape[1] // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


def main():
    args = parse_args()

    network, yolo_layers = load_network(args.config, args.weights)
    labels = load_labels(args.labels)
    if 'person' not in labels:
        sys.exit('Error: "person" class not found in {}'.format(args.labels))
    person_class = labels.index('person')

    # Box colour picked the same way as the tutorial (seeded random colour per class)
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    colour_box = [int(j) for j in colours[person_class]]

    # Open input video and create writer with the same fps and frame size
    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        sys.exit('Error: could not open video {}'.format(args.video))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

    mot_tracker = Sort()      # SORT multi-object tracker
    # SORT numbers tracks 1, 2, 3, ... in order of creation, so the highest
    # track ID returned so far is the number of people counted
    total_people = 0
    frame_number = 0
    total_time = 0.0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_number += 1
        start = time.time()

        detections = detect_people(network, yolo_layers, frame, person_class)
        tracks = mot_tracker.update(detections)   # rows of [x1, y1, x2, y2, track_id]
        if len(tracks) > 0:
            total_people = max(total_people, int(tracks[:, 4].max()))

        draw_tracks(frame, tracks, total_people, colour_box)
        writer.write(frame)

        frame_time = time.time() - start
        total_time += frame_time
        print('Frame {}: {:.4f} s'.format(frame_number, frame_time))

    video.release()
    writer.release()

    print('\nTotal frames processed: {}'.format(frame_number))
    if frame_number > 0:
        print('Average time per frame: {:.4f} s'.format(total_time / frame_number))
    print('Total number of people counted: {}'.format(total_people))
    print('Output video saved to: {}'.format(args.output))


if __name__ == '__main__':
    main()
