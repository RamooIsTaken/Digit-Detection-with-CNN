import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle


def preProcess(img) :
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)  # Histogramı 0-255 arası genişledi
    img = img / 255 # normalize 

    return img




batchSize = 250
path = "Data"
myList = os.listdir(path)
numOfClass = len(myList)

images = []
classNo = []

for i in range(numOfClass) :
    myImageLıst = os.listdir(path+"\\"+str(i))

    for j in myImageLıst :
        img = cv2.imread(path+"\\"+str(i)+"\\"+str(j))
        img = cv2.resize(img,(32,32))
        images.append(img)
        classNo.append(i)


#print(len(images),len(classNo)) #verinin dorğu yüklendiğini doğrulamak için totalde 10160 resim var 


images = np.array(images)
classNo = np.array(classNo)

# print(images.shape,classNo.shape)

#veri ayırma

xTrain,xTest,yTrain,yTest = train_test_split(images,classNo,test_size=0.5,random_state=42)
xTrain,xValidation,yTrain,yValidation = train_test_split(xTrain,yTrain,test_size=0.2,random_state=42)

# print(images.shape,testX.shape,validationX.shape)

fig,axes = plt.subplots(3, 1, figsize=(7,7))
fig.subplots_adjust(hspace=0.5)
sns.countplot(yTrain,ax=axes[0])
axes[0].set_title("TrainY")

sns.countplot(yTest,ax=axes[1])
axes[1].set_title("testY")

sns.countplot(yValidation,ax=axes[2])
axes[2].set_title("ValidationY")
plt.show()


#preprocess
xTrain = np.array(list(map(preProcess,xTrain)))
xTest = np.array(list(map(preProcess,xTest)))
xValidation = np.array(list(map(preProcess,xValidation)))

xTrain = xTrain.reshape(-1,32,32,1) 
xTest = xTest.reshape(-1,32,32,1)
xValidation = xValidation.reshape(-1,32,32,1)


#data generate 
dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.1,
                             rotation_range = 10)

dataGen.fit(xTrain)
#trainGen = dataGen.flow(trainX, trainY, batch_size=batchSize)

yTrain = to_categorical(yTrain,numOfClass)
yTest = to_categorical(yTest,numOfClass)
yValidation = to_categorical(yValidation,numOfClass)

model = Sequential()
model.add(Conv2D(input_shape=(32,32,1),
                 filters=8,
                 kernel_size=(5,5),
                 activation="relu",
                 padding="same"# piksel (1 sıra ) ekleme
                 ))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,
                 kernel_size=(3,3),
                 activation="relu",
                 padding="same"# piksel (1 sıra ) ekleme
                 ))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2)) #overfitting engeli için
model.add(Flatten())
model.add(Dense(units=256,
                activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=numOfClass,
                activation="softmax")) # çıkış katmanı

model.compile(loss="categorical_crossentropy",
              optimizer=("Adam"),
              metrics=["accuracy"])

#stepsPerEpoch = trainX.shape[0]//batchSize

hist = model.fit(dataGen.flow(xTrain,yTrain,batch_size=batchSize),
                validation_data=(xValidation,yValidation),
                epochs = 20,
                steps_per_epoch=xTrain.shape[0]//batchSize,
                shuffle=1)

"""pickleOut = open("trainedModel.p","wb")
pickle.dump(model,pickleOut)
pickleOut.close()"""


#Değerlendirme - Görselleştirme

# hist.history.keys = "val_loss","vall_accuracy","loss","accuracy" verilen çıktılar

plt.figure()
plt.plot(hist.history["loss"],label="Eğitim Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()
plt.show()


plt.figure()
plt.plot(hist.history["accuracy"],label="Eğitim accuracy")
plt.plot(hist.history["val_accuracy"],label="Validation accuracy")
plt.legend()
plt.show()


score = model.evaluate(xTest,yTest,verbose=1) #verbose görselleştirme için
print("Test Loss :",score[0])
print("Test accuracy :",score[1])



yPred = model.predict(xValidation) #tahmin yapıldı
yPredClass = np.argmax(yPred,axis=1) # değeri max bulunanın indeksi,
yTrue = np.argmax(yValidation,axis=1) 

cm = confusion_matrix(yTrue,yPredClass)
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm,
            annot=True, # koordinat değerleninin gözükmesi
            linewidths=0.01, #line kalınlığı
            cmap="Greens", #Yeşillerden oluşsun manasında 
            linecolor="gray", # Aradaki çizgilerin rengi
            fmt=".1f", # Virgülden sonra 1 basamak olsun
            ax=ax)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# Test veri seti üzerinde modeli değerlendirme
score = model.evaluate(xTest, yTest, verbose=1)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])


# Validation veri seti üzerinde tahmin yapma
yPred = model.predict(xValidation)
yPredClass = np.argmax(yPred, axis=1)
yTrue = np.argmax(yValidation, axis=1)

# Sınıflandırma doğruluğunu hesaplama
classificationAccuracy = np.mean(yPredClass == yTrue)
print("Sınıflandırma Doğruluğu:", classificationAccuracy)

# Confusion matrix'i hesaplama
cm = confusion_matrix(yTrue, yPredClass)

precision = precision_score(yTrue, yPredClass, average='macro')

recall = recall_score(yTrue, yPredClass, average='macro')

f1 = f1_score(yTrue, yPredClass, average='macro')


print("Hassasiyet (Precision):", precision)
print("Duyarlılık (Recall):", recall)
print("F1-Skoru:", f1)



