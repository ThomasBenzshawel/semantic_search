import pymongo
import os
import time
import certifi
import inference
import torch

ca = certifi.where()

uri = 'mongodb+srv://gatywill:Z86Qe7qi5qkR1dbd@vectors.bigshoc.mongodb.net/?retryWrites=true&w=majority&appName=Vectors'
client = pymongo.MongoClient(uri, server_api=pymongo.server_api.ServerApi('1'))

data_will_pc_path = 'C:\\Users\\wgaty\\Documents\\School\\Spring 23-24\\NLP\\Term Project Local\\Documents\\sparknotes\\literature'

db = client['corpus']
collection = db['sparknotes_lit']

dir_list = os.listdir(data_will_pc_path)
idx = 1

for filename in dir_list:
    path = os.path.join(data_will_pc_path, filename)
    f = open(path, 'r', encoding='utf-8', errors='ignore')
    fname_split = ''.join(filename.split('.')[:-1]).split("__")[1:]
    if fname_split[-1] == '':
        fname_split = fname_split[:-1]
    topic = fname_split[0]
    title = fname_split[1]
    section = "NA"
    if len(fname_split) >= 3:
        section = fname_split[2]
    subsection = "NA"
    if len(fname_split) >= 4:
        subsection = fname_split[3]
    txt = f.read()
    if not txt == '':
        vectorized_doc = inference.vectorize_document(txt)
        doc = {"filename": filename, "topic": topic, "title": title, "section": section, "subsection": subsection, "text": txt, "vector": vectorized_doc.tolist()}
        collection.insert_one(doc)
        print(f'\'{filename}\' ({idx} / {len(dir_list)})')
    else:
        print(f'skipped empty doc \'{filename}\' ({idx} / {len(dir_list)})')
    idx += 1
    time.sleep(0.025)

# for doc in collection.find():
#     # path = os.path.join(data_will_pc_path, filename)
#     # f = open(path, 'r', encoding='utf-8', errors='ignore')
#     # fname_split = ''.join(filename.split('.')[:-1]).split("__")[1:]
#     # if fname_split[-1] == '':
#     #     fname_split = fname_split[:-1]
#     # topic = fname_split[0]
#     # title = fname_split[1]
#     # section = "NA"
#     # if len(fname_split) >= 3:
#     #     section = fname_split[2]
#     # subsection = "NA"
#     # if len(fname_split) >= 4:
#     #     subsection = fname_split[3]
#     # doc = {"filename": filename, "title": title, "section": section, "subsection": subsection, "text": f.read()}
#     #doc = {"filename": filename, "topic": topic, "title": title, "section": section, "subsection": subsection, "text": f.read()} #totally forgot to add the topic field on the first runthrough... oops
#     vectorized_doc = inference.vectorize_document(doc["text"])
#     # print(vectorized_doc.tolist())
#     collection.find_one_and_update({"filename": doc["filename"]}, {'$set': {"vector": vectorized_doc[0].tolist()}})
#     print(doc["filename"], vectorized_doc)
#     time.sleep(0.03)



