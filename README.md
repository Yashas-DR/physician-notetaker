# 🏥 Physician Notetaker NLP Pipeline

A comprehensive AI-powered system for processing clinical transcripts between physicians and patients, providing automated medical documentation, sentiment analysis, and SOAP note generation.

## 🎯 Project Overview

This project implements a complete Natural Language Processing pipeline that transforms raw medical conversation transcripts into structured, actionable medical documentation. The system leverages advanced NLP techniques to extract medical entities, analyze patient sentiment, and generate professional medical reports.

## ✨ Features

### 🔍 **Medical Entity Recognition & Summarization**
- **Symptom Extraction**: Automatically identifies and extracts symptoms from patient descriptions
- **Diagnosis Detection**: Recognizes medical conditions and diagnostic information
- **Treatment Identification**: Captures treatment plans, medications, and therapeutic interventions
- **Prognosis Analysis**: Extracts recovery timelines and outcome predictions
- **Keyword Extraction**: Identifies relevant medical terminology and phrases

### 🧠 **Sentiment & Intent Analysis**
- **Patient Sentiment Classification**: Categorizes patient emotional states as `Anxious`, `Neutral`, or `Reassured`
- **Intent Detection**: Identifies conversational intentions such as:
  - Reporting symptoms
  - Seeking reassurance
  - Expressing concern
  - Describing treatment
  - Clarifying recovery

### 📄 **SOAP Note Generation**
- **Structured Documentation**: Generates professional SOAP notes following medical standards
- **Subjective**: Patient-reported information and complaints
- **Objective**: Clinical observations and examination findings
- **Assessment**: Medical diagnosis and condition evaluation
- **Plan**: Treatment recommendations and follow-up instructions

## 🚀 Quick Start

### Prerequisites
```bash
# Required Python packages
pip install nltk flask pandas numpy
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd physician-notetaker-nlp

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

### Running the Application

#### 1. Command Line Interface
```bash
python physician_notetaker.py
```

#### 2. Web Application
```bash
python flask_app.py
```
Then open your browser to `http://localhost:5000`

#### 3. API Usage
```python
from physician_notetaker import PhysicianNotetaker

# Initialize the system
notetaker = PhysicianNotetaker()

# Process a transcript
results = notetaker.process_transcript(your_transcript)

# Get specific outputs
medical_summary = notetaker.generate_medical_summary(transcript)
sentiment_analysis = notetaker.analyze_patient_sentiment_intent(transcript)
soap_note = notetaker.generate_soap_note(transcript)
```

## 📊 Sample Output

### Medical Summary
```json
{
  "Patient_Name": "Patel",
  "Symptoms": ["swollen", "sore", "pain", "stiff"],
  "Diagnosis": ["grade 2 sprain"],
  "Treatment": ["rest", "ice", "crutches", "brace", "ibuprofen", "physiotherapy"],
  "Current_Status": "healing well",
  "Prognosis": "recover fully in about 4–6 weeks",
  "Key_Phrases": ["ankle", "football", "injury", "recovery", "treatment"]
}
```

### Sentiment & Intent Analysis
```json
[
  {
    "Patient_Statement": "I twisted my ankle pretty badly last weekend while playing football, and it's still swollen and sore.",
    "Sentiment": "Neutral",
    "Intent": "Reporting symptoms"
  },
  {
    "Patient_Statement": "That's good to know. When can I start walking without crutches?",
    "Sentiment": "Reassured",
    "Intent": "Seeking reassurance"
  }
]
```

### SOAP Note
```json
{
  "Subjective": {
    "Chief_Complaint": "I twisted my ankle pretty badly last weekend while playing football, and it's still swollen and sore.",
    "History_of_Present_Illness": "It was last Saturday, around 4 PM. I was playing a friendly match..."
  },
  "Objective": {
    "Physical_Exam": "Let me do a quick physical exam to check your range of motion and stability.",
    "Observations": "Your ankle is healing well. There's some stiffness and mild swelling..."
  },
  "Assessment": {
    "Diagnosis": "grade 2 sprain",
    "Severity": "Moderate"
  },
  "Plan": {
    "Treatment": "You can gradually start putting more weight on it over the next week...",
    "Follow_Up": "If anything worsens or doesn't improve, come back in two weeks for a reassessment."
  }
}
```

## 🔗 API Endpoints

### Medical Summary
```bash
POST /api/medical-summary
Content-Type: application/json

{
  "transcript": "Your medical transcript here..."
}
```

### Sentiment & Intent Analysis
```bash
POST /api/sentiment-intent
Content-Type: application/json

{
  "transcript": "Your medical transcript here..."
}
```

### SOAP Note Generation
```bash
POST /api/soap-note
Content-Type: application/json

{
  "transcript": "Your medical transcript here..."
}
```

## 🛠️ Technical Implementation

### Core Technologies
- **Python 3.8+**: Primary programming language
- **NLTK**: Natural language processing toolkit
- **Flask**: Web framework for API and web interface
- **Regular Expressions**: Pattern matching for medical entities
- **JSON**: Structured data output format

### NLP Techniques Used
1. **Tokenization**: Breaking down text into meaningful units
2. **Part-of-Speech Tagging**: Identifying grammatical roles
3. **Named Entity Recognition**: Extracting medical entities
4. **Pattern Matching**: Rule-based extraction of medical information
5. **Sentiment Analysis**: Emotional state classification
6. **Intent Detection**: Purpose identification in conversations

### Medical Domain Adaptations
- **Medical Vocabulary**: Specialized dictionaries for symptoms, treatments, and conditions
- **Clinical Patterns**: Regex patterns for medical terminology
- **SOAP Structure**: Standard medical documentation format
- **Severity Classification**: Medical condition severity assessment

## 📁 Project Structure

```
physician-notetaker-nlp/
├── physician_notetaker.py      # Main NLP pipeline
├── flask_app.py               # Web application
├── templates/
│   ├── index.html            # Main web interface
├── requirements.txt          # Python dependencies
├── README.md                # This file
```

## 🧪 Testing

### Running Tests
```bash
# Test with sample transcript
python physician_notetaker.py

# Test web interface
python flask_app.py
# Open browser to http://localhost:5000/demo
```

### Sample Transcript Format
```
Physician: Good morning, Mr. Smith. What brings you in today?

Patient: I've been having chest pain for the past two days.

Physician: Can you describe the pain for me?

Patient: It's a sharp pain that comes and goes, especially when I breathe deeply.
```

## 🔧 Configuration

### Customizing Medical Dictionaries
Edit the dictionaries in `PhysicianNotetaker.__init__()`:
```python
self.medical_symptoms = {
    'pain', 'ache', 'sore', 'swollen', 'tender',
    # Add your custom symptoms here
}
```

### Adjusting Sentiment Thresholds
Modify sentiment analysis logic in `analyze_sentiment()` method to fine-tune classification accuracy.

## 📈 Performance Considerations

- **Processing Time**: ~2-5 seconds per transcript (depending on length)
- **Memory Usage**: Minimal - suitable for real-time processing
- **Scalability**: Can handle multiple concurrent requests via Flask
- **Accuracy**: Rule-based approach provides consistent, predictable results



## 📚 Medical Standards Compliance

This system follows established medical documentation standards:
- **SOAP Note Format**: Standardized clinical documentation
- **Medical Terminology**: Uses standard medical vocabulary
- **Privacy Considerations**: Designed for de-identified data processing
- **Clinical Workflow**: Integrates with existing medical documentation processes





---

**Note**: This system is designed for educational and research purposes. For clinical use, ensure compliance with relevant healthcare regulations and standards (HIPAA, GDPR, etc.).