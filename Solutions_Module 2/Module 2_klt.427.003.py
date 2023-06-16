import cv2
import numpy as np

# Задаем путь к видеофайлу
video_path = "data/processing/klt.427.003.mp4"

# Создаем объект VideoCapture
cap = cv2.VideoCapture(video_path)

# Определяем размеры кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Определяем порог для поиска светлых участков
threshold = 220

while True:
    # Считываем кадр из видео
    ret, frame = cap.read()

    if ret:
        # Конвертируем кадр в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Находим светлые участки на изображении
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Строим область для коррекции вокруг найденных участков
        kernel = np.ones((3, 3), np.uint8)
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, dist_thresh = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)
        dist_thresh = cv2.dilate(dist_thresh, kernel, iterations=2)

        # Выполняем гамма-коррекцию для области по маске
        gamma = 0.5
        result = np.copy(frame)
        for c in range(frame.shape[2]):
            result[:, :, c] = np.power(frame[:, :, c] / 255.0, gamma) * 255.0

        # Заменяем на исходном изображении требующие коррекции области на скорректированные
        result[dist_thresh != 0] = frame[dist_thresh != 0]

        # Отображаем результат
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()