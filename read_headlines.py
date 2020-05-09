import json

artList = []
print("Started Reading JSON file which contains multiple JSON document")
with open('data/headlines/Sarcasm_Headlines_Dataset.json') as f:
    for jsonObj in f:
        art = json.loads(jsonObj)
        artList.append(art)

print("Printing each JSON Decoded Object")
for art in artList[0:10]:
    print(art["headline"], art["is_sarcastic"])
