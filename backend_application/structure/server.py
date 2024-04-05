from flask import Flask, Response, request, send_file, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
import socketio
import pymongo
import sys
import inference

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
sio = socketio.Client()

socket_io = SocketIO(app, cors_allowed_origins="*", logger=True)
socket_io.init_app(app, cors_allowed_origins="*")

#API private key: 4606caf8-81da-4abe-9ea8-a3b4759b2a1a

uri = 'mongodb+srv://default_user:6J383XfBONlfrx1r@vectors.bigshoc.mongodb.net/?retryWrites=true&w=majority&appName=Vectors'
client = pymongo.MongoClient(uri, server_api=pymongo.server_api.ServerApi('1'))
db = client['corpus']
collection = db['sparknotes_lit']

USER_TEXT = ""

@app.route('/user_input', methods=['POST'])
def user_input():
    # global user_text
    global USER_TEXT
    USER_TEXT = request.get_json(force=True).get('input')
    return 'Success'

@app.route('/upload_doc', methods=['POST'])
def upload_doc():
    # global user_text
    json = request.get_json(force=True)
    fname = json.get('filename')
    ftext = json.get('filecontents')
    db_upload(fname, ftext)
    return 'Success'

def db_search_query(vector, n_neighbors=10):
    return collection.aggregate([{"$vectorSearch": {
            "index": "vector_index",
            "path": "vector",
            "queryVector": vector,
            "numCandidates": n_neighbors,
            "limit": 1,
        }}])

def db_upload(filename, filecontents):
    print(filename)
    print(filecontents)
    vectorized_doc = inference.vectorize_document(filecontents)
    print(vectorized_doc.tolist())
    doc = {"filename": filename, "topic": "NA", "title": "NA", "section": "NA", "subsection": "NA", "text": filecontents, "vector": vectorized_doc.tolist()}
    collection.insert_one(doc)

@app.route('/result_text', methods=['GET'])
@cross_origin()
def return_output_text():
    result_vector = None
    global USER_TEXT
    if USER_TEXT is not None:
        result_vector = inference.predict(USER_TEXT)
    result = db_search_query(result_vector.tolist())
    doc = result.next()
    print(doc['title'])
    json = {"result_title": doc['filename'], "result_text": str(doc['text'])}
    print(jsonify(json).get_data())
    response = jsonify(json)
    return response


if __name__ == "__main__":
    socket_io.run(app, host='0.0.0.0', port=5000, debug=True)