import "./App.css"
import { useState, useEffect } from 'react';


// export default function App() {
export default function App() {
    let backendURL = "http://localhost:5000" //flask app
    let [text] = useState('')
    let [resultText, setResultText] = useState('Result document text will appear here')
    let [resultTitle, setResultTitle] = useState('Result document title will appear here')
    let [resultFile, setResultFile] = useState('')
    let resultText2 = ''
    // let  = "Result document text will appear here"


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
        // getResultFile()
    }

    // useEffect(() => {
    //     setResult("Result")
    // })

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
            // console.log(response)
        }).catch(e => console.log(e))
    }

    function getResultFile() {
        fetch(`${backendURL}/result_file`, {
            method: 'GET',
            mode: 'no-cors',
            headers: {
                'Content-Type': 'application/json'
            },
        }).then(response => response).then(data => {
            const reader = new FileReader()
            reader.readAsText(data)
            reader.onloadend = () => setResultFile(reader.result)
        }).catch(e => console.log(e))
    }

    const process = (event) => {
        text = event.target.value
    }

    return (
        <div className="App">
            <div className="main-container">
                <h1>LowEffortReactApp.js</h1>
            </div>
            <div className="main-container">
                <input type="text" className="form-control" placeholder="Enter Text" onChange={e => process(e)}/>
                <button type="submit" onClick={sendUserInput}>Enter</button>
            </div>
            <div className="main-container">
                <h2>Document Title</h2>
            </div>
            <div className="main-container">
                <h3>{resultTitle}</h3>
            </div>
            <div className="main-container">
                <h2>Document Text</h2>
            </div>
            <div className="main-container">
                <p>{resultText}</p>
            </div>
        </div>
    );
}