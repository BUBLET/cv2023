import cv2
import numpy as np

# открываем видеофайл для чтения
cap = cv2.VideoCapture('data/processing/trm.168.091.avi')

# проходим по каждому кадру видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # применяем фильтр увеличения резкости
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp_frame = cv2.filter2D(frame, -1, kernel)

    cv2.imshow('Sharp Video', sharp_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()