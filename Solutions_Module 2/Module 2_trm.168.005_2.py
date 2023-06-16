import cv2
import numpy as np

# Открываем видеофайл
cap = cv2.VideoCapture("data/processing/trm.168.005.avi")

# Проверяем, успешно ли открылся видеофайл
if not cap.isOpened():
    print("Не удалось открыть видеофайл")
    exit()

# Получаем ширину и высоту видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создаем объекты для работы с предыдущим и текущим кадрами
prev_frame = None
curr_frame = None

# Обрабатываем каждый кадр видео
frame_count = 0
while True:
    # Читаем текущий кадр
    ret, curr_frame = cap.read()

    # Проверяем, успешно ли прочитался текущий кадр
    if not ret:
        break
    
    # Пропускаем первый кадр
    if curr_frame is None:
        continue

    # Преобразуем кадры в оттенки серого
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Вычисляем разницу между предыдущим и текущим кадрами
        diff = cv2.absdiff(curr_gray, prev_gray)

        # Нормализуем разницу и применяем пороговое преобразование
        norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(norm_diff, 30, 255, cv2.THRESH_BINARY)

        # Инвертируем пороговое изображение, чтобы дворник стал прозрачным
        inv_thresh = cv2.bitwise_not(thresh)

        # Накладываем инвертированное пороговое изображение на текущий кадр
        result = cv2.bitwise_and(curr_frame, curr_frame, mask=inv_thresh)

        # Создаем окно с возможностью изменения размера
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

        # Отображаем текущий кадр в окне
        cv2.imshow('frame', result)

        # Если нажата клавиша 'q', выходим из цикла
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Сохраняем текущий кадр как предыдущий для следующей итерации
    prev_frame = curr_frame.copy()
    frame_count += 1

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()