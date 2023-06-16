#'data/processing/trm.174.007.avi'

import cv2

# Открыть видеофайл
cap = cv2.VideoCapture('data/processing/trm.174.007.avi')

# Установить значение гаммы
gamma = 0.4

# Применить гамма-коррекцию к каждому кадру видео
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Применить гамма-коррекцию
        frame = cv2.pow(frame / 255.0, gamma)
        frame = frame * 255.0
        frame = frame.astype('uint8')
        
        # Отобразить кадр
        cv2.imshow('frame', frame)
        
        # Выход из цикла по нажатию клавиши 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Освободить ресурсы
cap.release()
cv2.destroyAllWindows()