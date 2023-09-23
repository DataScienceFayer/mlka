import cv2
import pyrealsense2 as rs
import numpy as np

class MaskRCNN:
    def __init__(self):
        # Загрузка Mask RCNN
        self.net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                                 "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Генерация разных цветов для маски
        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))

        # Задаем пороги для обнаружения объектов
        self.detection_threshold = 0.7
        self.mask_threshold = 0.3

        self.classes = []
        with open("dnn/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

    def detect_objects_mask(self, bgr_frame):
        blob = cv2.dnn.blobFromImage(bgr_frame, swapRB=True)
        self.net.setInput(blob)

        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])

        # Детектор объектов
        frame_height, frame_width, _ = bgr_frame.shape
        detection_count = boxes.shape[2]

        obj_boxes = []
        obj_classes = []
        obj_contours = []

        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]
            color = self.colors[int(class_id)]
            if score < self.detection_threshold:
                continue

            # Координаты бокса объекта
            x = int(box[3] * frame_width)
            y = int(box[4] * frame_height)
            x2 = int(box[5] * frame_width)
            y2 = int(box[6] * frame_height)
            obj_boxes.append([x, y, x2, y2])

            # Добавляем класс
            obj_classes.append(class_id)

            # Контуры
            # Получаем координаты маски
            # Получаем маску
            mask = masks[i, int(class_id)]
            roi_height, roi_width = y2 - y, x2 - x
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            obj_contours.append(contours)

        return obj_boxes, obj_classes, obj_contours

    def draw_object_mask(self, bgr_frame, obj_boxes, obj_classes, obj_contours):
        for box, class_id, contours in zip(obj_boxes, obj_classes, obj_contours):
            x, y, x2, y2 = box
            roi = bgr_frame[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape
            color = self.colors[int(class_id)]

            roi_copy = np.zeros_like(roi)

            for cnt in contours:
                cv2.drawContours(roi, [cnt], -1, (int(color[0]), int(color[1]), int(color[2])), 3)
                cv2.fillPoly(roi_copy, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                roi = cv2.addWeighted(roi, 1, roi_copy, 0.5, 0.0)
                bgr_frame[y: y2, x: x2] = roi

        return bgr_frame

    def draw_object_info(self, bgr_frame, obj_boxes, obj_classes):
        for box, class_id in zip(obj_boxes, obj_classes):
            x, y, x2, y2 = box

            color = self.colors[int(class_id)]
            color = (int(color[0]), int(color[1]), int(color[2]))

            class_name = self.classes[int(class_id)]
            cv2.rectangle(bgr_frame, (x, y), (x + 250, y + 70), color, -1)
            cv2.putText(bgr_frame, class_name.capitalize(), (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)
            cv2.rectangle(bgr_frame, (x, y), (x2, y2), color, 1)

        return bgr_frame

# Открываем видеопоток с веб-камеры
cap = cv2.VideoCapture(0)

mask_rcnn = MaskRCNN()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    obj_boxes, obj_classes, obj_contours = mask_rcnn.detect_objects_mask(frame)
    frame = mask_rcnn.draw_object_mask(frame, obj_boxes, obj_classes, obj_contours)
    frame = mask_rcnn.draw_object_info(frame, obj_boxes, obj_classes)

    cv2.imshow("Mask RCNN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
