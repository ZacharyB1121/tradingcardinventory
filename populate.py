from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image
import os
import redis
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import TextField, VectorField

#Vector dimension for VGG16, it's how many values are in our output vector
# How did I know this? I just have VGG16 spit out a vector and check the len() of it
VECTOR_DIM = 25088
IMAGE_BASE = "./mtg_images"

#Load the VGG16 model
nn = VGG16(weights='imagenet',  include_top=False)
redis_conn = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)

# Let's find all the images we previously downloaded
all_card_images = []
for root, dirs, files in os.walk(IMAGE_BASE):
    for file in files:
        if file.endswith(".jpg"):
            all_card_images.append(os.path.join(root, file))


current = 0
total = len(all_card_images)
#Loop through them to put them in the DB
for card_image in all_card_images:
    current += 1
    if current < 2900:
        continue
    if current % 100 == 0:
        print(f"Processing image {current} of {total}   {current/total * 100}% ...")
    #Load the image and resize it to 224x224, this is the only size VGG16 can take
    img = image.load_img(card_image, target_size=(224, 224))
    img = image.img_to_array(img)
    #Keras, the toolkit we're using, needs to preprocess the image.
    x = preprocess_input(np.expand_dims(img.copy(), axis=0))
    #Finnaly run it through VGG16 to get a vector
    preds = nn.predict(x, verbose=None)
    #Reformat the vector so it's in a good format
    vector = preds.flatten()
    #Convert the vector to a format we can store well
    embed = np.array(vector, dtype=np.float32).tobytes()
    #Build the Unique ID back up for the card
    card_set = card_image.split(os.path.sep)[-2]
    card_num = card_image.split(os.path.sep)[-1].split(".")[0]

    key = f"i:{card_set}:{card_num}"
    #And store it!
    redis_conn.hset(key, mapping = {
        "set": card_set,
        "num": card_num,
        "content_vector": embed
        }
    )



# Now we ask Redis to Index our data
card_set = TextField('set')
card_num = TextField('num')
embedding = VectorField("content_vector",
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": "COSINE",
        "INITIAL_CAP": len(all_card_images),
    }
)
fields = [card_set, card_num, embedding]

redis_conn.ft("mtg_cards").create_index(
    fields=fields,
    definition = IndexDefinition(prefix=["i"], index_type=IndexType.HASH)
    )