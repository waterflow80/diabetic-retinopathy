# Prepare the model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


images = []
labels = []
# Loading the dataset into ndarray
dataset2 = DRDataset(img_dir=train_images_dir, df=train_df, img_dim=img_dim)
for i in range(train_size):
    images.append(dataset[i])
    labels.append(train_df.iloc[i, 1])
x_train = np.array(images)
x_train.shape

num_classes = 5
batch_size = 128
epochs = 24
img_rows, img_cols = 512, 512
img_dim = (img_rows, img_cols)
train_size = len(train_df)

x_train1 = x_train.reshape(train_size, img_rows, img_cols, 3)
print(f"x_train1.shape = {x_train1.shape}")
x_train1 = x_train1.astype('float32')
x_train1 /= 255
y_train = keras.utils.to_categorical(labels, num_classes)
input_shape = x_train1.shape[1]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
         activation='relu',
         input_shape=(img_rows, img_cols, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])


hist = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_train, y_train)
                )

score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# graphique
epoch_list = list(range(1, len(hist.history['accuracy']) + 1))
plt.plot(epoch_list, hist.history['accuracy'])
plt.plot(epoch_list, hist.history['val_accuracy'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.show()