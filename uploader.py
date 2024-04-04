import pymongo
import os
import time
import certifi

ca = certifi.where()

uri = 'mongodb+srv://gatywill:ppK6MgtG4HeorutX@vectors.bigshoc.mongodb.net/?retryWrites=true&w=majority&appName=Vectors'
client = pymongo.MongoClient(uri, server_api=pymongo.server_api.ServerApi('1'))

data_will_pc_path = 'C:\\Users\\wgaty\\Documents\\School\\Spring 23-24\\NLP\\Term Project Local\\Documents\\sparknotes\\literature'

db = client['corpus']
collection = db['sparknotes_lit']

for filename in os.listdir(data_will_pc_path):
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
    doc = {"filename": filename, "title": title, "section": section, "subsection": subsection, "text": f.read()}
    #doc = {"filename": filename, "topic": topic, "title": title, "section": section, "subsection": subsection, "text": f.read()} #totally forgot to add the topic field on the first runthrough... oops
    collection.insert_one(doc)
    print(filename)
    time.sleep(0.03)



