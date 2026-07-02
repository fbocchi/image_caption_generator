from collections import Counter
import json

import numpy as np

import tensorflow as tf
from tensorflow.data import Dataset
import keras.layers


def load_captions() -> dict[str, dict[str, list[str]]]:
    with open('Flickr8k/captions/captions.json') as f:
        return json.load(f)

def extract_captions(captions: dict[str, dict[str, list[str]]], split_name: str) -> dict[str, list[str]]:
    return captions[split_name].copy()

def extract_train_captions(captions: dict[str, dict[str, list[str]]]) -> list[str]:
    train_split = extract_captions(captions, "train")
    train_captions = []
    for image in train_split:
        train_captions.extend(train_split[image])
    return train_captions

def add_start_and_end_tokens(captions: list[str]) -> list[str]:
    return [ "[START] " + caption + " [END]" for caption in captions]

def build_vocab(captions: list[str]):
    freq_counter = Counter()

    for caption in captions:
        freq_counter.update(caption.split())

    vocab = [word for word, _ in freq_counter.most_common()]

    return vocab

def compute_max_caption_len(captions: list[str]):
    captions = [caption.split() for caption in captions]
    lengths = [len(caption) for caption in captions]
    return max(lengths)

def create_vectorizer(vocab: list[str], max_caption_len: int):
    vectorizer = keras.layers.TextVectorization(
        max_tokens=None,
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=max_caption_len,
        pad_to_max_tokens=False,
        vocabulary=vocab.copy(),
    )
    return vectorizer

def save_config(vectorizer: keras.layers.TextVectorization, vocab_size: int, max_caption_len: int, path: str):
    config = {
        "vec_config": vectorizer.get_config(),
        "vocab_size": vocab_size,
        "max_caption_len": max_caption_len
    }

    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def define_model(vocab_size: int, max_caption_len: int) -> keras.Model:

    image_features_input = keras.Input(shape=(2048,), name='image_features_input')
    image_features = keras.layers.Dropout(0.5, name="image_features_dropout")(image_features_input)
    image_features = keras.layers.Dense(256, activation='relu', name='image_features_dense')(image_features)

    caption_input = keras.Input(shape=(max_caption_len,), name='caption_input')
    caption_embedding = keras.layers.Embedding(vocab_size, 256, mask_zero=True, name='caption_embedding')(caption_input)
    caption_embedding = keras.layers.Dropout(0.5, name='caption_embedding_dropout')(caption_embedding)
    output = keras.layers.LSTM(256, name='lstm')(caption_embedding)

    merged_representation = keras.layers.add([image_features, output])
    merged_representation = keras.layers.Dense(256, activation='relu', name='merge_dense')(merged_representation)
    vocab_distribution = keras.layers.Dense(vocab_size, activation='softmax', name='classifier')(merged_representation)

    model = keras.Model(inputs=[image_features_input, caption_input], outputs=vocab_distribution)
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam')

    print(model.summary())

    # plot_model(model, to_file='model.png', show_shapes=True)

    return model

def load_precomputed_image_features() -> dict[str, np.ndarray]:
    return np.load("flickr8k_resnet50v2_2048_features.npy", allow_pickle=True).item()

def create_datasets(
        image_features: dict[str, np.ndarray],
        captions: dict[str, dict[str, list[str]]],
        vectorizer: keras.layers.TextVectorization
) -> tuple[Dataset, Dataset]:

    train_images = load_images_by_split("train")
    val_images = load_images_by_split("val")

    train_image_features = extract_image_features(image_features, train_images)
    val_image_features = extract_image_features(image_features, val_images)

    train_captions = extract_captions(captions, "train")
    val_captions = extract_captions(captions, "val")

    train_set = create_dataset(train_image_features, train_captions, vectorizer, shuffle=True)
    val_set = create_dataset(val_image_features, val_captions, vectorizer)

    return train_set, val_set

def load_images_by_split(split_name: str) -> set[str]:
    with open(f"Flickr8k/captions/{split_name}_images.txt") as f:
        return {line.strip() for line in f}

def extract_image_features(image_features: dict[str, np.ndarray], image_names: set[str]) -> dict[str, np.ndarray]:
    split = {}
    for image in image_features:
        if image in image_names:
            split[image] = image_features[image]
    return split

def create_dataset(
        image_features: dict[str, np.ndarray],
        captions: dict[str, list[str]],
        vectorizer: keras.layers.TextVectorization,
        shuffle=False
) -> Dataset:

    input_captions = []
    target_words = []
    image_features_list = []

    for image, image_captions in captions.items():

        for caption in image_captions:
            caption = f"[START] {caption} [END]"
            tokens = caption.split()

            for i in range(1, len(tokens)):
                input_caption_tokens = tokens[:i]
                input_caption = " ".join(input_caption_tokens)
                target_word = tokens[i]

                input_captions.append(input_caption)
                target_words.append(target_word)
                image_features_list.append(image)

    input_captions = vectorizer(input_captions).numpy()

    vocab = vectorizer.get_vocabulary()
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    unk = word_to_idx.get("[UNK]", 1)

    target_words = np.array(
        [word_to_idx.get(w, unk) for w in target_words],
        dtype=np.int32
    )

    image_features_array = np.array([
        image_features[name]
        for name in image_features_list
    ], dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "image_features_input": image_features_array,
                "caption_input": input_captions,
            },
            target_words
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(target_words), reshuffle_each_iteration=True)

    dataset = dataset.batch(64)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def create_callbacks():

    return [
        # 1) checkpoint per resume training
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/icg_epoch_{epoch:02d}.weights.h5",
            save_weights_only=True,
            save_freq="epoch",
            verbose=1
        ),

        # 2) best model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/best.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2
        )
    ]

def train_model(
        model: keras.Model,
        training_set: Dataset,
        validation_set: Dataset,
        start_epoch=0,
        total_epochs=30
) -> tf.keras.callbacks.History:

    history = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=total_epochs,
        initial_epoch=start_epoch,
        callbacks=create_callbacks(),
        verbose=1
    )

    return history

def save_model_history(history: tf.keras.callbacks.History, path: str):
    with open(path, "w") as f:
        json.dump(history.history, f)

def generate_test_captions(icg: keras.Model):
    pass

def main():
    captions = load_captions()
    train_caption  = extract_train_captions(captions)
    train_captions = add_start_and_end_tokens(train_caption)
    vocab = build_vocab(train_captions)
    max_caption_len = compute_max_caption_len(train_captions)
    vectorizer = create_vectorizer(vocab.copy(), max_caption_len)
    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab)
    save_config(vectorizer, vocab_size, max_caption_len, "config.json")
    icg = define_model(vocab_size, max_caption_len) # teacher forcing!
    image_features = load_precomputed_image_features()
    training_set, validation_set = create_datasets(image_features, captions, vectorizer)
    history = train_model(icg, training_set, validation_set)
    save_model_history(history, "history.json")

if __name__ == '__main__':
    main()