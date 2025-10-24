import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay


def get_class_names(train_dir):
    """
    Gibt eine alphabetisch sortierte Liste der Klassenbezeichnungen zurück,
    basierend auf den Unterordnern im angegebenen Trainingsverzeichnis.
    """
    return sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

def FFN(ordner, model_name, epochen, n_full, droprate=0.0, resize=None, padding=True, val_ordner=None, aug_parameter=None, alpha=0, lr=0.001, decay=True):
    droprate=0
    aug_parameter=None
    alpha=0
    train_dir = ordner + '/'
    val_dir = val_ordner
    val_set = None

    # irgendeine Datei aus dem Ordner nehmen
    sample_path = os.path.join(train_dir, os.listdir(train_dir)[0],
                               os.listdir(os.path.join(train_dir, os.listdir(train_dir)[0]))[0])
    with Image.open(sample_path) as img:
        img_width, img_height = img.size  # PIL gibt (Breite, Höhe)
    print(f"Ermittelte Bildgröße der Trainingsdaten: {img_height}x{img_width}")

    if val_ordner is not None:
        sample_path = os.path.join(val_dir, os.listdir(val_dir)[0],
                                   os.listdir(os.path.join(val_dir, os.listdir(val_dir)[0]))[0])
        with Image.open(sample_path) as img:
            img_width2, img_height2 = img.size  # PIL gibt (Breite, Höhe)
        print(f"Ermittelte Bildgröße der Validierungsdaten: {img_height2}x{img_width2}")


    # Bildparameter
    img_height = img_height
    img_width = img_width
    batch_size = 32

    if resize==None:
        # Trainings- und Validierungsdatensätze erstellen
        dataset = image_dataset_from_directory(
            train_dir,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            label_mode='categorical'  # Mehrklassen-Klassifikation
        )
        num_classes = len(dataset.class_names)
        print("Datensatz in Originalgröße verarbeitet")
        if val_ordner is not None:
            if padding == False:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden ohne Padding von {img_height2}x{img_width2} auf {img_height}x{img_width} skaliert")
            elif padding == True:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(img_height2, img_width2),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )

                def resize_with_padding(image, label):
                    image = tf.image.resize_with_pad(image, img_height, img_width)
                    return image, label

                val_set = val_set.map(resize_with_padding)
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden mit Padding von {img_height2}x{img_width2} auf {img_height}x{img_width} skaliert")

    elif resize[0] is not None:
        if padding==False:
            dataset = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                seed=123,
                image_size=(resize[0], resize[1]),  # hartes Resize
                batch_size=batch_size,
                label_mode='categorical'
            )
            img_height=resize[0]
            img_width=resize[1]
            num_classes = len(dataset.class_names)
            print("Datensatz resized ohne Padding ", resize)
            if val_ordner is not None:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(resize[0], resize[1]),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden ohne Padding von {img_height2}x{img_width2} auf {resize[0]}x{resize[1]} skaliert")

        elif padding==True:
            dataset = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                seed=123,
                image_size=(img_height, img_width),  # Originalgröße
                batch_size=batch_size,
                label_mode='categorical'
            )
            img_height=resize[0]
            img_width=resize[1]
            print("Datensatz resized mit padding ", resize)

            def resize_with_padding(image, label):
                image = tf.image.resize_with_pad(image, resize[0], resize[1])
                return image, label

            num_classes = len(dataset.class_names)
            dataset = dataset.map(resize_with_padding)
            if val_ordner is not None:
                val_set = image_dataset_from_directory(
                    val_dir,
                    seed=123,
                    image_size=(img_height2, img_width2),
                    batch_size=batch_size,
                    label_mode='categorical'  # Mehrklassen-Klassifikation
                )

                def resize_with_padding(image, label):
                    image = tf.image.resize_with_pad(image, resize[0], resize[1])
                    return image, label

                val_set = val_set.map(resize_with_padding)
                if img_height==img_height2 and img_width==img_width2:
                    print("Validierungsdaten in Originalgröße verarbeitet")
                else:
                    print(f"Validierungsdaten wurden mit Padding von {img_height2}x{img_width2} auf {resize[0]}x{resize[1]} skaliert")

    else:
        raise ValueError("Wert von resize muss die Form [Höhe,Breite] haben")


    # Dataset normalisieren
    AUTOTUNE = tf.data.AUTOTUNE

    # Normalisieren
    normalization_layer = layers.Rescaling(1. / 255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y),
                          num_parallel_calls=AUTOTUNE)
    if val_ordner is not None:
        val_set = val_set.map(lambda x, y: (normalization_layer(x), y),
                              num_parallel_calls=AUTOTUNE)

    if aug_parameter is not None:
        # Augmentation nur fürs Training
        data_augmentation = tf.keras.Sequential(
            [
                layers.RandomFlip(aug_parameter[0]),
                layers.RandomRotation(aug_parameter[1]),
                layers.RandomZoom(aug_parameter[2]),
                layers.RandomContrast(aug_parameter[3]),
            ],
            name="data_augmentation",
        )

        print("Augmentation aktiv")
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=AUTOTUNE)

    # Pipeline tunen
    dataset = dataset.prefetch(AUTOTUNE)

    initial_lr = lr  # z. B. 1e-3
    card = tf.data.experimental.cardinality(dataset).numpy()  # -1=INFINITE, -2=UNKNOWN
    if card <= 0:
        # Fallback: wenn dir train_samples und batch_size bekannt sind, nimm:
        # steps_per_epoch = math.ceil(train_samples / batch_size)
        raise ValueError("steps_per_epoch konnte nicht bestimmt werden. Bitte angeben oder Fallback setzen.")
    steps_per_epoch = int(card)
    decay_steps = epochen * steps_per_epoch  # Gesamtanzahl Schritte
    dec = 1e-5 / initial_lr
    lr_schedule = CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=dec
    )

    if val_ordner is not None:
        val_set = val_set.prefetch(AUTOTUNE)

    # CNN-Modell definieren
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(img_height, img_width, 3)))

    for i in range(len(n_full)):
        model.add(layers.Dense(n_full[i], kernel_regularizer=regularizers.l2(alpha), use_bias=False, kernel_initializer='he_normal'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(droprate))

    model.add(layers.Dense(num_classes, activation='softmax'))

    # Modellkompilierung
    if decay is True:
        model.compile(optimizer=Adam(learning_rate=lr_schedule),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print("Mit lr-decay")
    else:
        model.compile(optimizer=Adam(learning_rate=lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


    if val_ordner is not None:
        history = model.fit(dataset, validation_data=val_set, epochs=epochen)
    else:
        history = model.fit(dataset, epochs=epochen)

    if val_set is not None:
        results = model.evaluate(val_set, verbose=0)
        print(results)
    # Modell speichern
    model.save(model_name)
    print(model.summary())

def validation_classification(model, val_ordner, padding=True):
    # --- Bild-Params (wie im Training) ---
    sample_path = os.path.join(val_ordner, os.listdir(val_ordner)[0],
                               os.listdir(os.path.join(val_ordner, os.listdir(val_ordner)[0]))[0])
    with Image.open(sample_path) as img:
        img_width, img_height = img.size  # PIL gibt (Breite, Höhe)
    print(f"Ermittelte Bildgröße der Validierungsbilder: {img_height}x{img_width}")

    # --- Modell laden ---
    model = keras.models.load_model(model)
    print(f"Ermittelte Bildgröße des Modells: {model.input_shape[1]}x{model.input_shape[2]}")
    img_size = (model.input_shape[1], model.input_shape[2])
    batch_size = 32

    if padding==False:
        val_ds = keras.utils.image_dataset_from_directory(
            val_ordner,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False,
            label_mode="int"
        )
        class_names = val_ds.class_names
        print(f"Größe Bilder im Datensatz: {img_height}x{img_width} vs. im Modell: {img_size}")
    elif padding==True:
        val_ds = keras.utils.image_dataset_from_directory(
            val_ordner,
            image_size=(img_height,img_width),
            batch_size=batch_size,
            shuffle=False,
            label_mode="int"
        )
        class_names = val_ds.class_names
        print(f"Datensatz von {img_height}x{img_width} resized mit padding auf {img_size}")

        def resize_with_padding(image, label):
            image = tf.image.resize_with_pad(image, img_size[0], img_size[1])
            return image, label

        val_ds = val_ds.map(resize_with_padding)

    print("Klassen (aus val_daten abgeleitet):", class_names)

    # --- gleiche Vorverarbeitung wie im Training: Resize + Rescaling(1/255) ---
    normalization = keras.layers.Rescaling(1. / 255)
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    # --- Vorhersagen einsammeln ---
    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- Metriken ---
    acc = accuracy_score(y_true, y_pred)
    f1_weight = f1_score(y_true, y_pred, average="weighted")
    f1_per_cls = f1_score(y_true, y_pred, average=None)  # array in Klassenreihenfolge

    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"F1 (weighted):       {f1_weight:.4f}")
    print("\nF1 pro Klasse:")
    for name, f1c in zip(class_names, f1_per_cls):
        print(f"  {name:>12}: {f1c:.4f}")
    return acc