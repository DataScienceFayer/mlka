import cv2
from mask_rcnn import MaskRCNN

# Создаем экземпляр класса MaskRCNN
mrcnn = MaskRCNN()

# Открываем веб-камеру (замените 0 на номер вашей веб-камеры, если у вас есть несколько)
cap = cv2.VideoCapture(0)

while True:
    # Получение кадра в реальном времени с веб-камеры
    ret, bgr_frame = cap.read()

    if not ret:
        print("Ошибка при чтении кадра с веб-камеры.")
        break

    # Получение маски объектов
    boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

    # Отрисовка маски объектов
    bgr_frame = mrcnn.draw_object_mask(bgr_frame)

    # Отображение информации о найденных объектах
    mrcnn.draw_object_info(bgr_frame, None)  # Вместо depth_frame передаем None

    # Отображение кадра с маской объектов
    cv2.imshow('Webcam Mask', bgr_frame)

    key = cv2.waitKey(1)
    if key == 27:  # Нажмите ESC, чтобы выйти из цикла
        break

# Освобождаем ресурсы и закрываем окна
cap.release()
cv2.destroyAllWindows()
