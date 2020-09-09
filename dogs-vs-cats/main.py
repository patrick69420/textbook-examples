import os, shutil
from keras import layers, models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

def create_small_set():
    this_dir = os.getcwd()
    data_dir = os.path.join(this_dir, 'data')
    small_dir = os.path.join(data_dir, 'small')
    if not os.path.isdir(small_dir):
        os.mkdir(small_dir)
    small_dirs = ['train', 'validation', 'test']
    for folder in small_dirs:
        dir = os.path.join(small_dir, folder)
        if not os.path.isdir(dir):
            os.mkdir(dir)
            os.mkdir(os.path.join(dir, 'cats'))
            os.mkdir(os.path.join(dir, 'dogs'))

    train_dir = os.path.join(data_dir, 'train')

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join(os.path.join(small_dir, 'train/cats'),
                           fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join(os.path.join(small_dir, 'validation/cats'),
                           fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join(os.path.join(small_dir, 'test/cats'),
                           fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join(os.path.join(small_dir, 'train/dogs'),
                           fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join(os.path.join(small_dir, 'validation/dogs'),
                           fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(train_dir, fname)
        dst = os.path.join(os.path.join(small_dir, 'test/dogs'),
                           fname)
        shutil.copyfile(src, dst)

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def main():
    create_small = False
    if create_small:
        create_small_set()
    small_dir = os.path.join(os.getcwd(), 'data/small')
    train_dir = os.path.join(small_dir, 'train')
    validation_dir = os.path.join(small_dir, 'validation')

    print('training cat images:', len(os.listdir(os.path.join(small_dir, 'train/cats'))))
    print('training dog images:', len(os.listdir(os.path.join(small_dir, 'train/dogs'))))
    print('validation cat images:', len(os.listdir(os.path.join(small_dir, 'validation/cats'))))
    print('validation dog images:', len(os.listdir(os.path.join(small_dir, 'validation/dogs'))))
    print('test cat images:', len(os.listdir(os.path.join(small_dir, 'test/cats'))))
    print('test dog images:', len(os.listdir(os.path.join(small_dir, 'test/dogs'))))

    model = create_model()
    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # resize all images to 150 x 150 (somewhat arbitrary choice)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    for data_batch, labels_batch in train_generator:
        print('data batch shape:', data_batch.shape)
        print('labels batch shape:', labels_batch.shape)
        break

    # steps_per_epoch=100 b/c batch_size=20 and there are 2000 train images
    # validation_steps=50 b/c batch_size=20 and there are 1000 validation images
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
        )

    model.save('cats_and_dogs_small_1.h5')

    breakpoint()


if __name__ == "__main__":
    main()