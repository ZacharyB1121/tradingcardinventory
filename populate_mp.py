#from pymilvus import MilvusClient
#import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import json
import time
import requests
import numpy as np
from keras.preprocessing import image
import os
from multiprocessing import Process, Queue, Value
import redis

IMAGE_BASE = "./mtg_images"

#print(tf.config.list_physical_devices())
num_cores = int(os.cpu_count() * 1) #1 = 52, 0.5 = 57, 1.5 = 60
print("Number of cores:", num_cores)


def worker(in_q, counter):
  pid = str(os.getpid())

  print(f"Worker [{pid}] Hello!")
  nn = VGG16(weights='imagenet',  include_top=False)
  print(f"Worker [{pid}] Loaded model!")
  #redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)
  redis_conn = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)
  worker_c = 0

  
  print(f"Worker [{pid}] Off we go!")
  while True:
    img_path = in_q.get()
    if img_path == None:
      break

    #w_time = time.time()
    with counter.get_lock():
      counter.value += 1
    #print(f"Worker [{pid}] INC - "+str(time.time() - w_time))
    #w_time = time.time()
    worker_c += 1
    if worker_c % 100 == 0:
      print(f"Worker [{pid}] Processed {worker_c} cards")

    img = image.load_img(img_path, target_size=(224, 224))
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
    card_set = img_path.split(os.path.sep)[-2]
    card_num = img_path.split(os.path.sep)[-1].split(".")[0]

    key = f"i:{card_set}:{card_num}"
    #And store it!
    redis_conn.hset(key, mapping = {
        "set": card_set,
        "num": card_num,
        "content_vector": embed
        }
    )

if __name__ == '__main__':
  counter = Value('i', 0)
  in_q = Queue()
  total = 0
  start_time = time.time()

  worker_count = num_cores * 0
  if worker_count == 0:
    worker_count = 8
  print("Worker count:", worker_count)

  print("Starting workers...")
  for i in range(worker_count):
    p = Process(target=worker, args=(in_q, counter))
    p.start()

  print("Loading data...")

  all_card_images = []
  for root, dirs, files in os.walk(IMAGE_BASE):
    for file in files:
      if file.endswith(".jpg"):
        all_card_images.append(os.path.join(root, file))

  for card_image in all_card_images:
    in_q.put(card_image)

  total = len(all_card_images)

  last_c = 0
  print("Wating the counter...")
  while counter.value < total:
    time.sleep(1)
    c = counter.value
    
    if (c % 50 == 0 or c - last_c > 100) and c != 0:
      last_c = c
      time_remaining = (time.time() - start_time) / c * (total - c)
      print(f"Time elapsed: {time.time() - start_time}, Estimated time remaining: {time_remaining / 60} minutes {c} of {total} { str(c/total * 100)}%")

  print("All done!")