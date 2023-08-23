import cv2
import numpy as np
import pickle




def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


pickleIn = open("trainedModel1.p", "rb")
model = pickle.load(pickleIn)

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)

while True:
    success, frame = cap.read()

    
    img = np.asarray(frame)
    img = cv2.resize(img, (32, 32))
    img = preProcess(img)

    img = img.reshape(1, 32, 32, 1)

    
    predictions = model.predict(img)
    classIndex = np.argmax(predictions) #en yüksek olasılığa sahip değeri alır alır
    probVal = np.amax(predictions)  #max döndürür
    print(classIndex, probVal)

    if probVal > 0.38:
        cv2.putText(frame, f"{classIndex} {probVal}", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("Rakam Sınıflandırma", frame)
    

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
