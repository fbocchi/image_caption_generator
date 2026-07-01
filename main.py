# This is a sample Python script.
from collections import defaultdict

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def preprocess_dataset(path):
    import json
    import string

    # -----------------------------
    # Funzione di pulizia
    # -----------------------------
    def clean_caption(caption):
        # lowercase
        caption = caption.lower()

        # remove punctuation
        caption = caption.translate(str.maketrans("", "", string.punctuation))

        # split into words
        words = caption.split()

        # remove short words and words containing digits
        words = [
            word
            for word in words
            if len(word) > 1 and word.isalpha()
        ]

        return " ".join(words)

    # -----------------------------
    # Carica il JSON
    # -----------------------------
    with open("Flickr8k/captions/captions.json", "r") as f:
        data = json.load(f)

    # -----------------------------
    # Pulisci tutte le caption
    # -----------------------------
    for split in ("train", "val", "test"):
        for image, captions in data[split].items():
            data[split][image] = [
                clean_caption(caption)
                for caption in captions
            ]

    # -----------------------------
    # Salva il risultato
    # -----------------------------
    with open("Flickr8k/captions/captions_clean.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Pulizia completata.")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # assumi che esistano i file captions.json, train_images.txt, ecc.
    # Assume captions are alredy clean

    # define model(vocab_sise, sequence_length)

    # precompute image features

    # add <end> and <
    # compute vocab_size and sequence_length
    # create training set and val set (triplets (image id, input_seq, SINGLE target_token)
    # proviamo a usare tf.data
    # instatiate model

    # train model

    preprocess_dataset('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
