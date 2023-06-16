import cv2
import numpy as np

# Открываем видеофайл
cap = cv2.VideoCapture("data/processing/klt.544.006.mp4")

# Обрабатываем каждый кадр видео
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Пропускаем первый кадр
    if frame_count == 0:
        frame_count += 1
        continue
    
    # Применяем пороговое преобразование для выделения светлых участков
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)

    # Применяем преобразование расстояний для получения маски
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, mask = cv2.threshold(dist_transform, 0.0001*dist_transform.max(), 255, 0)

    # Применяем гамма-коррекцию для области по маске
    gamma = 0.5 
    masked_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_img = cv2.pow(masked_img/255.0, gamma)
    masked_img = np.uint8(masked_img*255)

    # Заменяем требующие коррекции области на скорректированные
    result = cv2.bitwise_and(frame, frame, mask=np.uint8(mask))

    # Отображаем текущий кадр в окне
    cv2.imshow('frame', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    frame_count += 1

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()