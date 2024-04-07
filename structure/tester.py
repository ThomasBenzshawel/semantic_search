import pymongo
import sys
import inference
import os
import run_trainer
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score


uri = 'mongodb+srv://default_user:6J383XfBONlfrx1r@vectors.bigshoc.mongodb.net/?retryWrites=true&w=majority&appName=Vectors'
client = pymongo.MongoClient(uri, server_api=pymongo.server_api.ServerApi('1'))
db = client['corpus']
collection = db['sparknotes_lit']

def run_test_on_pretrained(documents, labels, n_splits=5):
    kf = KFold(n_splits=n_splits)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for train_index, test_index in kf.split(documents):
        # Split the data into training and testing sets
        train_docs, test_docs = [documents[i] for i in train_index], [documents[i] for i in test_index]
        train_labels, test_labels = [labels[i] for i in train_index], [labels[i] for i in test_index]
        print(test_labels)
        
        # Perform inference on the test documents
        predicted_labels = []
        for doc in test_docs:
            
            vector = inference.predict(doc)
            # print(vector)
            # print(vector.tolist())
            result = db_search_query(vector.tolist())
            doc = result.next()
            predicted_label = doc['title']
            print(predicted_label)
            predicted_labels.append(predicted_label)
        
        # Calculate evaluation metrics
        precision = precision_score(test_labels, predicted_labels, average='weighted')
        recall = recall_score(test_labels, predicted_labels, average='weighted')
        f1 = f1_score(test_labels, predicted_labels, average='weighted')
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Calculate the average scores across all folds
    avg_precision = sum(precision_scores) / n_splits
    avg_recall = sum(recall_scores) / n_splits
    avg_f1 = sum(f1_scores) / n_splits
    
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F-measure: {avg_f1:.4f}")



def run_test_with_train(documents, labels, n_splits=5):
    kf = KFold(n_splits=n_splits)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for train_index, test_index in kf.split(documents):
        # Split the data into training and testing sets
        train_docs, test_docs = [documents[i] for i in train_index], [documents[i] for i in test_index]
        train_labels, test_labels = [labels[i] for i in train_index], [labels[i] for i in test_index]
        
        # Train the model on the training set
        model = run_trainer.train_preloaded_data(train_docs)

        predicted_labels = []
        for doc in train_docs:
            vector = inference.predict(doc, model)
            result = db_search_query(vector.tolist())
            doc = result.next()
            predicted_label = doc['title']
            print(predicted_label)
            predicted_labels.append(predicted_label)

        # Calculate evaluation metrics for the training set
        train_precision = precision_score(train_labels, predicted_labels, average='weighted')
        train_recall = recall_score(train_labels, predicted_labels, average='weighted')
        train_f1 = f1_score(train_labels, predicted_labels, average='weighted')

        print("Training set:")
        print(f"Precision: {train_precision:.4f}")
        print(f"Recall: {train_recall:.4f}")
        print(f"F-measure: {train_f1:.4f}")


        
        # Perform inference on the test documents
        predicted_labels = []
        for doc in train_docs:
            vector = inference.predict(doc, model)
            result = db_search_query(vector.tolist())
            doc = result.next()
            predicted_label = doc['title']
            print(predicted_label)
            predicted_labels.append(predicted_label)
        
        # Calculate evaluation metrics
        precision = precision_score(test_labels, predicted_labels, average='weighted')
        recall = recall_score(test_labels, predicted_labels, average='weighted')
        f1 = f1_score(test_labels, predicted_labels, average='weighted')

        print("Test set:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F-measure: {f1:.4f}")
        
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Calculate the average scores across all folds
    avg_precision = sum(precision_scores) / n_splits
    avg_recall = sum(recall_scores) / n_splits
    avg_f1 = sum(f1_scores) / n_splits

    avg_train_precision = sum(train_precision) / n_splits
    avg_train_recall = sum(train_recall) / n_splits
    avg_train_f1 = sum(train_f1) / n_splits

    print("Training set:")
    print(f"Precision: {avg_train_precision:.4f}")
    print(f"Recall: {avg_train_recall:.4f}")
    print(f"F-measure: {avg_train_f1:.4f}")


    print("Test set:")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F-measure: {avg_f1:.4f}")

def db_search_query(vector, n_neighbors=10):
    return collection.aggregate([{"$vectorSearch": {
            "index": "vector_index",
            "path": "vector",
            "queryVector": vector,
            "numCandidates": n_neighbors,
            "limit": 1,
        }}])
    

def test_on_folder(folder_path, pretrained=False):
    documents = []
    labels = []
    
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding="utf-8") as file:
            document = file.read()
            documents.append(document)
            #Split on __ to get the label
            print(filename)
            filename = filename.split('__')[2]
            filename = filename.split('.')[0]
            print(filename)
            labels.append(filename)  # The filename represents the label

        
    if pretrained:
        run_test_on_pretrained(documents, labels)
    else:
        run_test_with_train(documents, labels)

if __name__ == '__main__':
    folder_path = sys.argv[1]
    pretrained = sys.argv[2] == 'pretrained'
    test_on_folder(folder_path, pretrained)