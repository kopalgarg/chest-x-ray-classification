import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

batch_size = 256
num_classes = 3
epochs = 10

CONV1 = 64
CONV2 = 128
DENSE = 128
KERNEL_CONV=3
KERNEL_POOL=2
ACTIVATION='relu'
DROPOUT = 0.1
OPTIMIZER='adam'

#input image dimensions
img_rows, img_cols = 32, 32

model = Sequential()
model.add(Conv2D(CONV1, kernel_size=(KERNEL_CONV, KERNEL_CONV),
                 activation=ACTIVATION,
                 input_shape=(img_rows,img_cols,1)))
model.add(MaxPooling2D((KERNEL_POOL, KERNEL_POOL)))
model.add(Conv2D(CONV2, (KERNEL_CONV, KERNEL_CONV), activation='relu'))
model.add(Flatten())
model.add(Dense(DENSE, activation=ACTIVATION, name ='my_dense'))
model.add(Dropout(DROPOUT))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer = OPTIMIZER, metrics = ['accuracy'])


def average_performance_CNN(iter, X_train, Y_train, X_test, Y_test):
    acc= []
    prec = []
    rec = []
    f1 = []
    for i in range(iter):
        history = model.fit(X_train.reshape(X_train.shape[0],img_rows,img_cols,1),
                    tf.keras.utils.to_categorical(Y_train, num_classes),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.20)
        Y_pred = np.argmax(model.predict(X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)), axis=1)        
        acc.append(metrics.accuracy_score(Y_test, Y_pred))
        prec.append(metrics.precision_score(Y_test, Y_pred, average='macro'))
        rec.append(metrics.recall_score(Y_test,Y_pred, average='macro'))
        f1.append(metrics.f1_score(Y_test, Y_pred, average='macro'))

    m_acc, h_acc = mean_confidence_interval(acc, confidence=0.95)
    m_prec, h_prec = mean_confidence_interval(prec, confidence = 0.95)
    m_rec, h_rec = mean_confidence_interval(rec, confidence = 0.95)
    m_f1, h_f1 = mean_confidence_interval(f1, confidence=  0.95)

    print("Mean Accuracy: ", m_acc, "+-", h_acc)
    print("Mean Precision: ", m_prec, "+-", h_prec)
    print("Mean Recall: ", m_rec, "+-", h_rec)
    print("Mean F1: ", m_f1, "+-", h_f1)

average_performance(20, projected_X_train, Y_train, projected_X_test, Y_test)