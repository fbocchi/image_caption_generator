# This is a sample Python script.
from collections import defaultdict

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def preprocess_dataset(path):
    from collections import Counter

    # -----------------------------
    # Carica gli split
    # -----------------------------
    def load_images(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    train_images = load_images("Flickr8k/captions/train_images.txt")
    val_images = load_images("Flickr8k/captions/val_images.txt")
    test_images = load_images("Flickr8k/captions/test_images.txt")

    # -----------------------------
    # Controlla duplicati nello stesso file
    # -----------------------------
    for name, images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ]:
        counter = Counter(images)
        duplicates = [img for img, count in counter.items() if count > 1]

        print(f"{name}: {len(images)} immagini")

        if duplicates:
            print(f"  Duplicati ({len(duplicates)}):")
            for img in duplicates:
                print(f"    {img} ({counter[img]} volte)")
        else:
            print("  Nessun duplicato")

    # -----------------------------
    # Trasforma in set
    # -----------------------------
    train_set = set(train_images)
    val_set = set(val_images)
    test_set = set(test_images)

    # -----------------------------
    # Controlla sovrapposizioni
    # -----------------------------
    print("\nSovrapposizioni:")
    print("train ∩ val :", len(train_set & val_set))
    print("train ∩ test:", len(train_set & test_set))
    print("val ∩ test  :", len(val_set & test_set))

    # -----------------------------
    # Numero totale di immagini
    # -----------------------------
    all_images = train_set | val_set | test_set

    print("\nTotale immagini uniche:", len(all_images))

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
