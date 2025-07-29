# NLP Dashboard Application

A FastAPI-based web application for advanced text analysis and visualization of document corpora. This application processes PDF, DOCX, TXT, CSV, and JSON files to perform semantic analysis, sentiment analysis, topic extraction, and summarization, presenting results through an interactive dashboard.



## 🚀 Key Features

### Document Processing
- **Multi-format Support**: Handles PDF, DOCX, TXT, CSV, and JSON files
- **Batch Processing**: Upload multiple documents via ZIP files
- **Text Extraction**: Advanced parsing using PyMuPDF, python-docx, and pandas
- **Caching**: Intelligent text caching for improved performance

### NLP Pipeline
- **Sentiment Analysis**: Uses DistilBERT fine-tuned on SST-2 for accurate sentiment classification
- **Topic Modeling**: TF-IDF + NMF approach for diverse topic extraction
- **Text Summarization**: DistilBART-CNN model for structured summaries
- **Word Cloud Generation**: Frequency-based visualization with sentiment coloring

### Interactive Dashboard
- **Real-time Analysis**: Automatic processing with progress tracking
- **Interactive Visualizations**: Plotly-based charts and word clouds
- **Context Exploration**: Click on topics/words to see source excerpts
- **Responsive Design**: Modern UI optimized for desktop and mobile



## 📁 Project Structure

```
Analytic board/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration settings
├── log_store.py           # Global state management
├── requirements_info.txt   # Python dependencies
├── models/                # NLP pipeline components
│   ├── nlp_pipeline.py    # Main pipeline orchestration
│   ├── sentiment_analyzer.py  # DistilBERT sentiment analysis
│   ├── summarizer.py      # DistilBART summarization
│   └── topic_modeler.py   # TF-IDF + NMF topic extraction
├── routers/               # API endpoints
│   ├── upload.py          # File upload and processing
│   ├── analyze.py         # NLP pipeline execution
│   └── visualize.py       # Data formatting for frontend
├── utils/                 # Utility functions
│   ├── data_cleaner.py    # Text preprocessing
│   ├── file_parser.py     # ZIP file handling
│   └── text_extractor.py  # Multi-format text extraction
├── templates/             # Jinja2 HTML templates
│   ├── base.html          # Base template
│   ├── index.html         # Upload page
│   ├── dashboard.html     # Results dashboard
│   └── error.html         # Error pages
└── static/               # Frontend assets
    ├── css/
    │   └── styles.css     # Global styles
    └── js/
        ├── app.js         # Main application logic
        └── dashboard.js   # Dashboard interactivity
```



## 🛠️ Technology Stack

### Backend
- **FastAPI** (v0.109.2) - Modern web framework
- **Uvicorn** - ASGI server
- **Jinja2** - Template engine
- **Python 3.11+** - Core runtime

### NLP & ML
- **Transformers** (v4.37.2) - Hugging Face models
- **DistilBERT** - Sentiment analysis
- **DistilBART-CNN** - Text summarization
- **scikit-learn** - Topic modeling (TF-IDF + NMF)
- **NLTK** - Text preprocessing
- **PyMuPDF** - PDF processing
- **python-docx** - DOCX processing
- **pandas** - CSV/JSON processing

### Frontend
- **Plotly.js** - Interactive visualizations
- **Vanilla JavaScript** - Dashboard interactivity
- **CSS Grid/Flexbox** - Responsive layout
- **Modern CSS** - Clean, professional styling

### Document Processing
- **PyMuPDF** (fitz) - PDF text extraction
- **python-docx** - DOCX document parsing
- **pandas** - CSV data processing
- **json** - JSON file handling



## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git (for cloning)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Analytic-board
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements_info.txt
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Run the Application
```bash
python main.py
```

The application will be available at `http://127.0.0.1:8000`



## 📖 Usage Guide

### 1. Upload Documents
- Navigate to the home page
- Prepare a ZIP file containing your documents (PDF, DOCX, TXT, CSV, JSON)
- Click "Choose File" and select your ZIP
- Click "Upload ZIP" to start processing

### 2. View Analysis Results
- After upload, you'll be automatically redirected to the dashboard
- The dashboard displays:
  - **Key Threats**: Identified risks and vulnerabilities
  - **Current Landscape**: Overview of findings
  - **Defense Strategies**: Recommended solutions
  - **Topic Analysis**: Interactive bar chart of key topics
  - **Word Cloud**: Frequency and sentiment visualization

### 3. Explore Context
- Click on any topic in the bar chart to see source excerpts
- Click on words in the word cloud for detailed context
- Modal windows show relevant paragraphs with sentiment scores

### 4. Processing Logs
- View real-time processing steps and any errors
- Logs are displayed at the bottom of the dashboard

## 🔧 API Endpoints

### File Upload
- `POST /zip` - Upload and process ZIP files
- `GET /upload-status` - Check upload processing status

### Analysis
- `POST /run-pipeline` - Execute NLP pipeline
- `GET /analysis-status` - Check analysis status
- `GET /analysis-results` - Retrieve analysis results
- `DELETE /clear-results` - Clear stored results

### Visualization
- `GET /results` - Get formatted visualization data
- `GET /download-results` - Download analysis results

### Health & Status
- `GET /` - Home page
- `GET /dashboard` - Results dashboard
- `GET /health` - Health check endpoint

## 🧠 NLP Pipeline Details

### 1. Text Extraction
- **PDF**: PyMuPDF extracts text with layout preservation
- **DOCX**: python-docx processes Word documents
- **CSV**: pandas converts tabular data to text
- **JSON**: Native JSON parsing with text extraction
- **TXT**: Direct UTF-8 text reading

### 2. Text Preprocessing
- Whitespace normalization
- Duplicate removal
- Paragraph segmentation
- Basic cleaning and formatting

### 3. Sentiment Analysis
- **Model**: DistilBERT fine-tuned on SST-2
- **Input**: Individual paragraphs
- **Output**: Polarity scores (-1 to 1) and labels
- **Chunking**: Automatic text chunking for long paragraphs

### 4. Topic Modeling
- **Method**: TF-IDF + Non-negative Matrix Factorization (NMF)
- **Features**: 1-3 gram sequences
- **Diversity**: Similarity-based deduplication
- **Fallback**: N-gram extraction for small corpora

### 5. Summarization
- **Model**: DistilBART-CNN-12-6
- **Structure**: Key findings, implications, recommendations
- **Length**: Adaptive based on input size
- **Chunking**: Smart text splitting for large documents

### 6. Word Cloud Generation
- **Frequency**: TF-IDF-based term importance
- **Sentiment**: Color coding based on sentiment scores
- **Phrases**: Multi-word term extraction
- **Normalization**: Relative frequency scaling

## 🎨 Dashboard Features

### Interactive Visualizations
- **Topic Bar Chart**: Horizontal bars with sentiment coloring
- **Word Cloud**: Size = frequency, color = sentiment
- **Click Interactions**: Modal windows with context
- **Responsive Design**: Adapts to screen size

### Data Exploration
- **Context Search**: Find source paragraphs for any term
- **Sentiment Analysis**: Per-paragraph sentiment scores
- **Document Tracking**: Link topics to source documents
- **Export Capabilities**: Download results in JSON format

### User Experience
- **Real-time Processing**: Live progress updates
- **Error Handling**: Graceful error display
- **Loading States**: Visual feedback during processing
- **Mobile Responsive**: Works on all device sizes

## 🔍 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
ENVIRONMENT=development
MODEL_PATH=models/bert-base-uncased
```

### Application Settings
- **Max File Size**: Configurable upload limits
- **Processing Timeout**: Adjustable pipeline timeouts
- **Model Parameters**: Customizable NLP model settings
- **Cache Settings**: Text caching configuration

## 🧪 Testing

### Run Tests
```bash
pytest tests/
```

### Code Quality
```bash
black --check .
pytest --cov=app tests/
```

## 🚀 Deployment

### Development
```bash
python main.py
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Recommended)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_info.txt .
RUN pip install -r requirements_info.txt

COPY . .
RUN python -m spacy download en_core_web_sm

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for pre-trained models
- **FastAPI** for the excellent web framework
- **Plotly** for interactive visualizations
- **PyMuPDF** for PDF processing capabilities

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the processing logs for debugging
- Review the API documentation at `/docs` when running

---

**Note**: This application is optimized for processing financial crime and security-related documents, but can be adapted for any text analysis use case.


