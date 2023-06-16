import cv2
import numpy as np
# Загрузка видео
cap = cv2.VideoCapture('data/processing/trm.179.003.avi')


while(cap.isOpened()):
    # Считывание кадра
    ret, frame = cap.read()

    # Если кадр считан успешно
    if ret == True:
            # Применяем пороговое преобразование для выделения светлых участков
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

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


        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()