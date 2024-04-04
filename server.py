from flask import Flask, Response, request, send_file
from flask_socketio import SocketIO, emit
# from flask_cors import CORS
import socketio
import pymongo
import sys
# from structure import inference
import inference
#import Thing from '../structure/model.py'

app = Flask(__name__)
sio = socketio.Client()

socket_io = SocketIO(app, cors_allowed_origins="*", logger=True)
socket_io.init_app(app, cors_allowed_origins="*")

uri = 'mongodb+srv://gatywill:74hk0kkbMDhSAY1Y@vectors.bigshoc.mongodb.net/?retryWrites=true&w=majority&appName=Vectors'
client = pymongo.MongoClient(uri, server_api=pymongo.server_api.ServerApi('1'))
db = client['corpus']
collection = db['sparknotes_lit']

user_input = None

@app.route('/user_input', methods=['POST'])
def user_input():
    user_input = request.get_json(force=True).get('input')
    print(user_input)
    return 'Success'

def db_search_query(vector, n_neighbors=100):
    query = {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "", #dont think this can be implemented until the vectorized fields are added to the db
            "queryVector": vector,
            "numCandidates": n_neighbors,
            "limit": 1,
        }
    }
    return collection.find(query)

@app.route('/result_text', methods=['GET'])
def return_output_text():
    result_vector = None
    if user_input is not None:
        result_vector = inference.predict(user_input)
    result = db_search_query(result_vector)
    json = {"resultTitle": result['title'], "resultText": result['']}
    json['resultText'] = result
    return json


# @app.route('/result_file', methods=['GET']) #TODO fix once file return is implemented
# def return_output_file():
#     result = 'test'
#     if user_input is not None:
#         result = inference.predict(user_input)
#     json = {}
#     json['resultText'] = result
#     return result


if __name__ == "__main__":
    socket_io.run(app, host='0.0.0.0', port=5000, debug=True)