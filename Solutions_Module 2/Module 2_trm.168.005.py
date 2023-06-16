import cv2

# Задаем путь к видеофайлу
video_path = "data/processing/trm.168.005.avi"

# Создаем объект VideoCapture
cap = cv2.VideoCapture(video_path)

# Определяем размеры кадра
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создаем объект VideoWriter для сохранения результата
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result/trm_168_005_result.mp4', fourcc, 25.0, (width, height))

while True:
    # Считываем два поряд идущих кадра, начиная со второго
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    if ret:
        # Вычисляем разницу между кадрами
        diff = cv2.absdiff(frame1, frame2)

        # Нормализуем полученную разницу
        norm_diff = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Накладываем нормализованную разницу на текущий кадр
        result = cv2.addWeighted(frame1, 1, norm_diff, 0.5, 0)

        # Сохраняем результат
        out.write(result)

        # Отображаем результат
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()