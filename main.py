if __name__ == '__main__':

    # assumi che esistano i file captions.json, train_images.txt, ecc.
    # Assume captions are alredy clean
    # FATTO

    # definire delle funzioni per ciascuno step

    # 1) compute vocab_size and sequence_length

    # 2) define model(vocab_sise, sequence_length) (come su sito...)

    # 3) precompute image features
    # FATTO: codice in Colab coccobocch e features in Drive

    # add <start> ed <end>???

    # 4) create training set and val set (triplets (image id, input_seq, SINGLE target_token)
    # proviamo a usare tf.data

    # 5) train model