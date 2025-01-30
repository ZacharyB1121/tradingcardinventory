import requests
import json
import os
source_json = "./default-cards.json"

with open(source_json, encoding='utf-8') as f:
    json_data = json.load(f)

total = len(json_data)
print("Total cards: " + str(total))

current = 0

for card in json_data:
    current += 1
    card_name = card["name"]
    card_set = card["set"]

    if card_set != "m20":
        continue

    card_collectors_number = card["collector_number"]
    
    card_year = card["released_at"][0:4]

    if int(card_year) < 2019:
        continue
    

    print("Downloading image " + str(current) + " of " + str(total) + "   " + str(current/total * 100) +"% ...")
    if card["image_status"] == "missing":
        continue
    try:
        card_image = card["image_uris"]["normal"]
    except:
        card_image = card["card_faces"][0]["image_uris"]["normal"] # For double sided cards

    card_image_url = card_image
    card_image_path = "." + os.path.sep + "mtg_images" + os.path.sep + card_set + os.path.sep
    os.makedirs(os.path.dirname(card_image_path), exist_ok=True)

    card_file_name = card_collectors_number + ".jpg"

    print(card_image_path + card_file_name)
    print(card_image_url)
    response = requests.get(card_image_url)
    file = open(card_image_path + card_file_name , "wb")
    file.write(response.content)
    file.close()
    print("Downloaded " + card_name + " image")
    print("---------------------------------------------------")