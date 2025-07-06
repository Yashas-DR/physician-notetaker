#!/usr/bin/env python3
"""
Flask Web Application for Physician Notetaker NLP Pipeline
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
from physician_notetaker import PhysicianNotetaker

app = Flask(__name__)

# Initialize the notetaker
notetaker = PhysicianNotetaker()

@app.route('/')
def index():
    """Main page with transcript input form."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_transcript():
    """Process the transcript and return results."""
    try:
        transcript = request.form.get('transcript', '')
        
        if not transcript.strip():
            return jsonify({'error': 'Please provide a transcript'}), 400
        
        # Process the transcript
        results = notetaker.process_transcript(transcript)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/medical-summary', methods=['POST'])
def api_medical_summary():
    """API endpoint for medical summary only."""
    try:
        data = request.get_json()
        transcript = data.get('transcript', '')
        
        if not transcript.strip():
            return jsonify({'error': 'Please provide a transcript'}), 400
        
        summary = notetaker.generate_medical_summary(transcript)
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment-intent', methods=['POST'])
def api_sentiment_intent():
    """API endpoint for sentiment and intent analysis only."""
    try:
        data = request.get_json()
        transcript = data.get('transcript', '')
        
        if not transcript.strip():
            return jsonify({'error': 'Please provide a transcript'}), 400
        
        analysis = notetaker.analyze_patient_sentiment_intent(transcript)
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/soap-note', methods=['POST'])
def api_soap_note():
    """API endpoint for SOAP note generation only."""
    try:
        data = request.get_json()
        transcript = data.get('transcript', '')
        
        if not transcript.strip():
            return jsonify({'error': 'Please provide a transcript'}), 400
        
        soap_note = notetaker.generate_soap_note(transcript)
        return jsonify(soap_note)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint for Render
@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # For production (Render will use gunicorn)
    app.run(debug=False)