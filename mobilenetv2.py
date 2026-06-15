import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Seed global para reprodutibilidade
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 3

PATH_REAL_TRAIN = Path("dataset/train")
PATH_AUGMENTED  = Path("_datasets_aumentados_v_final")
PATH_VALIDATION = Path("dataset/valid")

def build_mobilenet_v2(input_shape=(128, 128, 3), num_classes=3):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

def evaluate_and_plot(model, val_ds, class_names, experiment_name):
    print(f"\n--- Avaliação Final: {experiment_name} ---")
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    # Normalização por linha: cada valor dividido pelo total real da classe
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Anotação combinada: valor absoluto + percentual (recall por célula)
    labels = np.array([
        [f"{v}\n({p:.1%})" for v, p in zip(row_v, row_p)]
        for row_v, row_p in zip(cm, cm_norm)
    ])

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm_norm,              # escala de cor pela proporção
        annot=labels,         # texto mostra absoluto + percentual
        fmt='',               # fmt vazio porque annot já é string
        cmap='Blues',
        vmin=0, vmax=1,       # escala de cor sempre 0–100%
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Matriz de Confusão - {experiment_name}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(f"2_matriz_confusao_{experiment_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()

def executar_experimentos():
    print("Carregando dataset de validação...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        PATH_VALIDATION,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    class_names = val_ds.class_names
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Baseline incluído explicitamente
    cenarios = {
        "A":        PATH_AUGMENTED / "A",
        "B":        PATH_AUGMENTED / "B",
        "Baseline": PATH_REAL_TRAIN,
    }

    for nome, path_treino in cenarios.items():
        print(f"\nINICIANDO TREINAMENTO - CENÁRIO: {nome}\n")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            path_treino,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=SEED
        )
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        tf.random.set_seed(SEED)  # reinicia seed antes de cada modelo
        model = build_mobilenet_v2()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[early_stop]
        )

        evaluate_and_plot(model, val_ds, class_names, nome)
        model.save(f"final_mobilenetv2_{nome.lower()}.h5")

if __name__ == "__main__":
    executar_experimentos()