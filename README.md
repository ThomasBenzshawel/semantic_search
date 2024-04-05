# Semantic Search Demo Application
Setup instructions:
1. In the main working directory, run "npx create-react-app application", and copy the two files from the "React App Files" directory into the new application/src directory, overwriting the two that exist in there already
2. In the new "application" directory, run "npm init -y"
    - You may need to also run "npm i use-file-picker"
3. In the "backend_application" directory, run "python pip install -r requirements.txt" (assuming you have Python configured correctly)

To run:
1. Run "npm start" in the "application" directory
2. Run the server.py file located in the "backend_application/structure" directory

System use (from assignment page):  
- How the user will feed the test data into the system: type something (doesn't matter how long) into the input text box and click "enter".
- How the output will be reported to the user: the filename (which includes the subject, topic, and any subtopics) and text of the document returned by the model and the subsequent search will be displayed on the application page.
- How to interpret the output: being a semantic search, the goal of the system is to return a document related to the input phrase, without the name (or any part of it) of the document necessarily being specified in the input. Since it is a notetaking aid, the application is trained and tested on data from sources of notes, with the objective of linking parts of the notes documents that the user is currently writing to other ones that already exist.