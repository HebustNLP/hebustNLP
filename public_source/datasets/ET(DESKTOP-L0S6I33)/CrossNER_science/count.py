import json

dataset_name = "science"
dataset_type = "train"
with open(dataset_name + "/" + dataset_type + ".json", "r", encoding="utf-8") as file:
    dataset = json.load(file)

print("number of samples: " + str(len(dataset)))

tag = []
for item in dataset:
    for entity in item["entities"]:
        if (entity["type"] in tag) == False:
            tag.append(entity["type"])

print("number of tags: " + str(len(tag)))
print(tag) 

# #'''
# ### transform the tags into natural language
for item in dataset:
    for entity in item["entities"]:
        if entity["type"] == "astronomicalobject":
            entity["type"] = "astronomical object"
        elif entity["type"] == "chemicalcompound":
            entity["type"] = "chemical compound"
        elif entity["type"] == "academicjournal":
            entity["type"] = "academic journal"
        elif entity["type"] == "chemicalelement":
            entity["type"] = "chemical element"
#         elif entity["type"] == "LOC":
#             entity["type"] = "location"
#         elif entity["type"] == "WEA":
#             entity["type"] = "weapon"
#         elif entity["type"] == "FAC":
#             entity["type"] = "facility"

with open(dataset_name + "/" + dataset_type + ".json", "w", encoding="utf-8") as file:
    json.dump(dataset, file, ensure_ascii=False)
#'''
'''
### devide if dev/test dataset cannot be found
trainset = dataset[:420000]
testset = dataset[420000:]
with open("train.json", "w", encoding="utf-8") as file:
    json.dump(trainset, file, ensure_ascii=False)
with open("dev.json", "w", encoding="utf-8") as file:
    json.dump(testset, file, ensure_ascii=False)
with open("test.json", "w", encoding="utf-8") as file:
    json.dump(testset, file, ensure_ascii=False)   
'''
