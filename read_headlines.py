import json


# reads items in json file as a list of dicts, where the entries are "article_link", "headline", "is_sarcastic"
artList = []
with open('data/headlines/Sarcasm_Headlines_Dataset.json') as f:
    for jsonObj in f:
        art = json.loads(jsonObj)
        artList.append(art)

# extract tweets as dict mapping tweet to sarcasm = 1, not sarcasm = 0 
tweetlist = {}
with open('data/twitter/sarcasm-dataset.txt', 'r') as fp:
	for twt in fp:
  		tweetlist[twt[0:len(twt)-2].strip()] = twt[-2]
print(tweetlist)
#TODO: get rid of html and other random junk strings