import json

# carica modello
# definisci funzioni opportune

# credo che si dovrà ricostruire il vectorizer da

from keras.layers import  TextVectorization

with open("config.json", "r") as f:
    config = json.load(f)

print(config)
vec = TextVectorization.from_config(config["vec_config"])
print(vec(["[START] dog running [END]"]))