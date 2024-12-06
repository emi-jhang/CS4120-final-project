import React, { useState } from 'react';
import axios from 'axios';
import './App.css'

function TextSimplifier() {
  // State for the input text and simplified text
  const [text, setText] = useState('');
  const [simplifiedText, setSimplifiedText] = useState('');

  // Handle input text being changed
  const handleTextChange = (event) => {
    setText(event.target.value);
  };

  // Handle simplify button being pressed
  const handleSimplify = async () => {
    // If input text box doesn't have text in it, tell the user to give text input
    if (text.trim() === '') {
      alert("Please enter text for simplification.");
      return;
    }

    // Make http request to backend and if successful, display the simplified text, else display an error alert
    try {
      const response = await axios.post("http://127.0.0.1:5000/simplify", { text });
      setSimplifiedText(response.data.simplified_text);
    } catch (error) {
      console.error("Error simplifying text:", error);
      alert("An error occurred while simplifying the text.");
    }
  };

  // Start speech recognition (voice input)
  const startVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window)) {
      alert('Your browser does not support speech recognition.');
      return;
    }

    const recognition = new window.webkitSpeechRecognition();
    recognition.lang = 'en-US'; 
    recognition.start();

    recognition.onresult = (event) => {
      const spokenText = event.results[0][0].transcript;
      console.log('You said:', spokenText);
      setText(spokenText); // Set the recognized text to the input field
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
    };
  };

  // Speak the simplified text (text-to-speech)
  const speakSimplifiedText = () => {
    if (!('speechSynthesis' in window)) {
      alert('Your browser does not support text-to-speech.');
      return;
    }

    const speech = new SpeechSynthesisUtterance(simplifiedText);
    speech.lang = 'en-US';
    speech.rate = 0.9; 
    speech.pitch = 1; 
    window.speechSynthesis.speak(speech);
  };

  return (
    <div className="container">
      <h1>Text Simplification</h1>
      <p className="description">
        Enter your text below and press "Simplify" to see a simpler version.
      </p>

      <div className="text-box-container">
        <div className="input-box">
          <textarea
            value={text}
            onChange={handleTextChange}
            placeholder="Enter text here"
            rows="10"
          />
          <i className="fa fa-microphone" onClick={startVoiceInput} title="Start Voice Input"></i>
        </div>

        <button onClick={handleSimplify}>Simplify</button>

        <div className="output-box">
          <textarea
            value={simplifiedText}
            placeholder="Simplified text will appear here"
            rows="10"
            readOnly
          />
          <i className="fa fa-volume-up" onClick={speakSimplifiedText} title="Read Aloud"></i>
        </div>
      </div>
    </div>
  );
}

export default TextSimplifier;
