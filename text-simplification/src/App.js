import React, { useState } from 'react';
import axios from 'axios';
import './App.css'

function TextSimplifier() {
  // State for the input text and simplified text
  const [text, setText] = useState('');
  const [simplifiedText, setSimplifiedText] = useState('');

  const handleTextChange = (event) => {
    setText(event.target.value);
  };

  const handleSimplify = async () => {
    if (text.trim() === '') {
      alert("Please enter text for simplification.");
      return;
    }

    try {
      const response = await axios.post("http://127.0.0.1:5000/simplify", { text });
      setSimplifiedText(response.data.simplified_text);
    } catch (error) {
      console.error("Error simplifying text:", error);
      alert(error)
      // alert("An error occurred while simplifying the text.");
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
    speech.rate = 1; // Speed of speech (1.0 is normal)
    speech.pitch = 1; // Pitch of speech (1.0 is normal)
    window.speechSynthesis.speak(speech);
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Text Simplification</h1>

      <p style={{ fontSize: '18px', marginBottom: '20px' }}>
        Enter your text below and press "Simplify" to see a simpler version.
      </p>

      <button onClick={startVoiceInput} style={{ marginBottom: '10px' }}>
        Start Voice Input
      </button>
      
      <textarea
        value={text}
        onChange={handleTextChange}
        placeholder="Enter text here"
        rows="10"
        style={{ width: '100%', marginBottom: '10px' }}
      />

      <button onClick={handleSimplify} style={{ display: 'block', margin: '10px 0' }}>
        Simplify
      </button>

      <textarea
        value={simplifiedText}
        placeholder="Simplified text will appear here"
        rows="10"
        style={{ width: '100%', marginTop: '20px' }}
        readOnly
      />

      <button onClick={speakSimplifiedText} style={{ display: 'block', margin: '10px 0' }}>
        Read Aloud
      </button>
    </div>
  );
}

export default TextSimplifier;
