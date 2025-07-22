# Document Chatbot
A chatbot application that allows users to interact with and query the contents of PDF documents using natural language. The project features a backend for document processing and a frontend for user interaction.

## 📸 Project Screenshots
These screenshots showcase the complete interface of the Document Chatbot project.

### Home page 
<div align="center">
  <img src="top.jpg" alt="Amazon Clone Top Section" width="800"/>
  <p><em>The landing page welcomes users and provides an overview of the Document Chatbot. Users can see options to upload documents or start chatting.</em></p>
</div>

## 🛠️ Technologies Used

- Python 3.8+
- NLTK (Natural Language Toolkit)
- Vector Database (for document embeddings)
- HTML, CSS, JavaScript (frontend)

## 🚀 Getting Started

1. Clone the repository:
```bash
https://github.com/Okay002/Document-Chatbot.git
```

2. Navigate to the project directory:
```bash
cd Dcoument-Chatbot
```


3. Set up Python environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
4. Install python dependencies:
``` bash
pip install -r requirements.txt
```
5. Download NLTK data (if required):
```bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

6. Run the frontend:
``` bash
cd frontend
python app.py
# If using React or similar:
npm install
npm start
```

## 📁 Project Structure

```
Document-Chatbot/
├── backend/           # Backend code (APIs, document processing, vector DB)
├── frontend/          # Frontend code (UI, static files)
├── vectordbdata/      # Vector database files
├── output/            # Output files (e.g., processed docs)
├── nltk_data/         # NLTK data for NLP tasks
├── venv/              # Python virtual environment
├── requirements.txt   # Python dependencies
├── sample.pdf         # Example PDF document
├── Sample_2.pdf       # Example PDF document
├── Sample 3.pdf       # Example PDF document
└── README.md          # Project documentation
```
## 🎯 Key Technical Features

- Upload and process PDF documents
- Store and search document embeddings using a vector database
- Query documents with natural language questions
- User-friendly web interface
- Modular backend and frontend structure


## 👏 Acknowledgments
NLTK for natural language processing.

## 📄 License
This project is licensed under the MIT License.


