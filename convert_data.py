import json
import os

cwd = os.getcwd()
data_path = cwd  + "/label_data"
json_file_paths = []
image_file_paths = []
for subdir, dirs, files in os.walk(cwd):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".json"):
            json_file_paths.append(filepath)
        if filepath.endswith(".png"):
            image_file_paths.append(filepath)

categories = []
for path in json_file_paths:
    with open(path) as data_file:
        data = json.load(data_file)
        if "shapes" in data and len( data["shapes"] ) > 0:
            shape = data["shapes"][0]
            categories.extend(shape["label"])

# removing duplicates from list
categories = list(set(categories))
print(categories)





