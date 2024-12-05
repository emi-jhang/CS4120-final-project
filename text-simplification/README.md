# Text Simplification Application

This is a text simplification application that consists of a **frontend** built with **React** and a **backend** implemented in **Python**. The frontend sends text input to the backend, which processes it and returns a simplified version based on a custom lexicon.

## Prerequisites

Before setting up the application, ensure you have the following installed:

- **[Node.js](https://nodejs.org/)** (for the frontend)
- **[Python](https://www.python.org/)** (for the backend)
- A Python virtual environment tool (like `venv`) is recommended for isolating dependencies.

## Setup Instructions

### 1. Frontend Setup (React)

1. Navigate to the project directory that this README.md is located in
2. Install the necessary dependencies by running: 
`npm install`
3. Start the frontend in development mode by running: 
`npm start`
This will launch the React development server, and the frontend will be available in your browser at **http://localhost:3000**. Any changes you make to the frontend code will automatically be reflected in the browser.

### 2. Backend Setup (Python)

1. Navigate to the src files where server.py and script.py are located
2. (Optional) Create a virtual environment to manage Python dependencies: 
`python -m venv venv` 
`source venv/bin/activate`
3. Install the backend dependencies listed in requirements.txt by running: 
`pip install -r requirements.txt`
4. Start the backend server by running: 
`python server.py`
The backend server will run at **http://localhost:5000**. The frontend will communicate with this server to simplify the text. 

## Application Workflow
1. Open the frontend in your browser at http://localhost:3000.
2. Enter a sentence or paragraph into the input field.
3. The frontend sends a request to the backend to simplify the text. 
4. The backend processes the text and returns a simplified version, which is displayed on the frontend. 

## Troubleshooting
1. Ensure that both the frontend and backend servers are running.
2. Inspect the application and navigate to the Network tab for any errors if the text is not being simplified correctly. 
