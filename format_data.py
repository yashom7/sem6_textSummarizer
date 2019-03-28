import json_lines
import pickle

head =[]
desc = []
keywords = []
with open('sample-2K.jsonl', 'rb') as f:
    for item in json_lines.reader(f):
        print(item['id'])
        title = item['title'].strip().replace('\n', '').replace('\ ', '')
        content = item['content'].strip().replace('\n', '').replace('\ ', '')
        head.append(title)
        desc.append(content)
        keywords.append(None)

data_tuple = (head, desc, keywords)

with open('tokens.pkl','wb') as fp:
    pickle.dump(data_tuple, fp)
