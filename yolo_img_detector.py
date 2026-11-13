import os
import sys
import time
import argparse
import cv2
import numpy as np

# Phase 1: YOLOv3 object detection on images

prob_min = 0.5
nms_thresh = 0.3


def get_args():
    parser = argparse.ArgumentParser(description='YOLOv3 object detection on images')
    parser.add_argument('-w', '--weights', required=True, help='yolov3.weights file')
    parser.add_argument('-c', '--config', required=True, help='yolov3.cfg file')
    parser.add_argument('-l', '--labels', required=True, help='coco.names file')
    parser.add_argument('-i', '--images', required=True, nargs='+', help='image(s) or folder of images')
    parser.add_argument('-o', '--out', default='out_imgs', help='output folder (default: out_imgs)')
    parser.add_argument('-inf', action='store_true', help='show average inference time')
    parser.add_argument('-classes_all', action='store_true', help='show total objects/classes detected')
    parser.add_argument('-breakdown', action='store_true', help='show count for each class')
    parser.add_argument('-per_image', action='store_true', help='show counts for each image')
    args = parser.parse_args()

    for f in [args.weights, args.config, args.labels]:
        if not os.path.isfile(f):
            parser.error('file not found: ' + f)

    # if no output flags are given, show everything
    if not (args.inf or args.classes_all or args.breakdown or args.per_image):
        args.inf = args.classes_all = args.breakdown = args.per_image = True

    return args, parser


def get_image_list(paths, parser):
    # turn the -i inputs (files and/or folders) into one list of images
    imgs = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    imgs.append(os.path.join(p, f))
        elif os.path.isfile(p):
            imgs.append(p)
        else:
            parser.error('path not found: ' + p)
    if len(imgs) == 0:
        parser.error('no images found')
    return imgs


def detect(net, yolo_layers, img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    start = time.time()
    net.setInput(blob)
    output = net.forward(yolo_layers)
    inf_time = time.time() - start

    h, w = img.shape[:2]
    boxes, confs, class_ids = [], [], []
    for result in output:
        for det in result:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = scores[cid]
            if conf > prob_min:
                cx, cy, bw, bh = (det[0:4] * np.array([w, h, w, h])).astype('int')
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                confs.append(float(conf))
                class_ids.append(cid)

    keep = cv2.dnn.NMSBoxes(boxes, confs, prob_min, nms_thresh)
    keep = np.array(keep).flatten() if len(keep) > 0 else []
    return [(class_ids[i], confs[i], boxes[i]) for i in keep], inf_time


def draw(img, dets, labels, colors):
    thick = max(1, img.shape[1] // 400)
    size = max(0.5, img.shape[1] / 1000)
    for cid, conf, (x, y, bw, bh) in dets:
        color = [int(c) for c in colors[cid]]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, thick)
        ty = y - 7 if y > 22 else y + 25  # keep text on screen
        cv2.putText(img, '{}: {:.4f}'.format(labels[cid], conf), (max(x, 0), ty),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, thick)


def main():
    args, parser = get_args()
    img_paths = get_image_list(args.images, parser)

    net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
    try:
        yolo_layers = net.getUnconnectedOutLayersNames()
    except AttributeError:
        yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

    with open(args.labels) as f:
        labels = [line.strip() for line in f if line.strip()]

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    os.makedirs(args.out, exist_ok=True)

    times = []
    total = {}       # class name -> count over all images
    per_image = []   # (image name, {class name: count})

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print('could not read ' + path + ', skipping')
            continue

        dets, t = detect(net, yolo_layers, img)
        times.append(t)

        counts = {}
        for cid, _, _ in dets:
            name = labels[cid].capitalize()
            counts[name] = counts.get(name, 0) + 1
            total[name] = total.get(name, 0) + 1
        per_image.append((os.path.basename(path), counts))

        draw(img, dets, labels, colors)
        name, ext = os.path.splitext(os.path.basename(path))
        cv2.imwrite(os.path.join(args.out, name + '_out' + ext), img)

    if len(times) == 0:
        sys.exit('no images could be read')

    if args.inf:
        print('Average Inference Time: {:.5f} seconds'.format(sum(times) / len(times)))
    if args.classes_all:
        print('Total Number of Objects Detected:', sum(total.values()))
        print('Total Number of Classes Detected:', len(total))
    if args.breakdown:
        print('\nTotal Detection Breakdown')
        for name in sorted(total):
            print('{}: {}'.format(name, total[name]))
    if args.per_image:
        print('\nPer Image Breakdown')
        for img_name, counts in per_image:
            s = ', '.join('{}: {}'.format(n, counts[n]) for n in sorted(counts))
            print(img_name, '=>', s if s else 'None')


if __name__ == '__main__':
    main()