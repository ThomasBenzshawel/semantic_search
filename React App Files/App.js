import "./App.css"
import { useState } from 'react';
import { useFilePicker } from 'use-file-picker';


export default function App() {
    let backendURL = "http://localhost:5000" //flask app
    let [text] = useState('')
    let [resultText, setResultText] = useState('Result document text will appear here')
    let [resultTitle, setResultTitle] = useState('Result document title will appear here')
    let [userInputFileName] = ''
    const { openFilePicker, filesContent, loading } = useFilePicker({
        accept: '.txt',
    });


    function sendUserInput(){
        console.log("called sendUserInput()")
        console.log(text)
        fetch(`${backendURL}/user_input`, {
            method: 'POST',
            mode: 'no-cors',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                input: text,
                confirm: "confirmed"
            })
        })
        getResultText()
    }

    function getResultText() {
        fetch(`${backendURL}/result_text`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            },
        }).then(response => response.json())
        .then((data) => {
            console.log(data)
            console.log(data.result_text)
            console.log(data.result_title)
            setResultText(data.result_text)
            setResultTitle(data.result_title)
        }).catch(e => console.log(e))
    }

    function uploadDocToDB(){
        fetch(`${backendURL}/upload_doc`, {
            method: 'POST',
            mode: 'no-cors',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: filesContent[0].path,
                filecontents: filesContent[0].content
            })
        })
    }

    const process = (event) => {
        text = event.target.value
    }

    return (
        <div className="App">
            <div className="sub-container">
                <h1>LowEffortReactApp.js</h1>
            </div>
            <div className="sub-container">
                <h2>User new file input demo:</h2>
            </div>
            <div className="sub-container">
                <button onClick={() => openFilePicker()}>Upload notes document here (.txt files only)</button>
                <br />
                {filesContent.map((file, index) => (
                    <br />
                ))}
            </div>
            <div>
                <button type="submit" onClick={uploadDocToDB}>Upload notes doc to DB</button>
            </div>
            <br />
            <br />
            <div className="sub=container">
                <h2>User text input demo:</h2>
            </div>
            <div className="sub-container">
                <input type="text" className="form-control" placeholder="Enter Text" onChange={e => process(e)}/>
                <button type="submit" onClick={sendUserInput}>Enter</button>
            </div>
            <div className="sub-container">
                <h3>Document Title</h3>
            </div>
            <div className="sub-container">
                <h4>{resultTitle}</h4>
            </div>
            <div className="sub-container">
                <h3>Document Text</h3>
            </div>
            <div className="sub-container">
                <p>{resultText}</p>
            </div>    
        </div>
    );
}