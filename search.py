from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing import image

import redis
from redis.commands.search.query import Query

#Load the VGG16 model
nn = VGG16(weights='imagenet',  include_top=False)
redis_conn = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)

img = image.load_img("McDo.jpg", target_size=(224, 224))
    
img = image.img_to_array(img)
x = preprocess_input(np.expand_dims(img.copy(), axis=0))
print("Sending image to model")
preds = nn.predict(x, verbose=None)
vector = preds.flatten()
#embed = np.array(vector, dtype=np.float32).tobytes()
embed = np.array(vector).astype(dtype=np.float32).tobytes()


return_fields = ["set", "num", "vector_score"]
base_query = f'*=>[KNN 20 @content_vector $vector AS vector_score]'

query = (
    Query(base_query)
        .return_fields(*return_fields)
        .sort_by("vector_score")
        .paging(0, 5)
        .dialect(2)
)

params_dict = {"vector": embed}
result = redis_conn.ft("mtg_cards").search(query, params_dict)

for i, article in enumerate(result.docs):
    score = 1 - float(article.vector_score)
    print(f"{i}. {article.set} {article.num} (Score: {round(score ,3) })")