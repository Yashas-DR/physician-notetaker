#!/usr/bin/env python3
"""
Physician Notetaker NLP Pipeline
A comprehensive system for processing clinical transcripts between physicians and patients.
"""

import re
import json
import nltk
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import pandas as pd
from datetime import datetime
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer

class PhysicianNotetaker:
    """
    Main class for processing medical transcripts and generating structured outputs.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Medical terminology and patterns
        self.medical_symptoms = {
            'pain', 'ache', 'sore', 'swollen', 'swelling', 'tender', 'stiff', 
            'numb', 'tingling', 'bruising', 'inflammation', 'discomfort'
        }
        
        self.medical_treatments = {
            'rest', 'ice', 'elevation', 'crutches', 'brace', 'physiotherapy',
            'therapy', 'medication', 'ibuprofen', 'surgery', 'exercise'
        }
        
        self.medical_conditions = {
            'sprain', 'fracture', 'strain', 'injury', 'whiplash', 'arthritis',
            'infection', 'diabetes', 'hypertension', 'pneumonia'
        }
        
        # Sentiment keywords
        self.anxiety_keywords = {
            'worried', 'concerned', 'scared', 'anxious', 'nervous', 'afraid',
            'terrible', 'awful', 'worse', 'worried'
        }
        
        self.reassurance_keywords = {
            'good', 'better', 'relief', 'glad', 'happy', 'thankful',
            'grateful', 'pleased', 'satisfied', 'confident'
        }
        
    def parse_transcript(self, transcript: str) -> Dict[str, List[str]]:
        """
        Parse the transcript to separate physician and patient statements.
        """
        lines = transcript.strip().split('\n')
        physician_statements = []
        patient_statements = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Physician:'):
                physician_statements.append(line.replace('Physician:', '').strip())
            elif line.startswith('Patient:'):
                patient_statements.append(line.replace('Patient:', '').strip())
        
        return {
            'physician': physician_statements,
            'patient': patient_statements
        }
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities using NLP techniques and medical dictionaries.
        """
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        # Extract symptoms
        symptoms = []
        for word in words:
            if word in self.medical_symptoms:
                symptoms.append(word)
        
        # Extract treatments
        treatments = []
        for word in words:
            if word in self.medical_treatments:
                treatments.append(word)
        
        # Extract conditions/diagnoses
        diagnoses = []
        for word in words:
            if word in self.medical_conditions:
                diagnoses.append(word)
        
        # Pattern-based extraction
        symptoms.extend(self._extract_symptom_patterns(text))
        treatments.extend(self._extract_treatment_patterns(text))
        diagnoses.extend(self._extract_diagnosis_patterns(text))
        
        return {
            'symptoms': list(set(symptoms)),
            'treatments': list(set(treatments)),
            'diagnoses': list(set(diagnoses))
        }
    
    def _extract_symptom_patterns(self, text: str) -> List[str]:
        """Extract symptoms using pattern matching."""
        patterns = [
            r'(sharp|dull|chronic|acute|severe|mild|moderate)\s+(pain|ache)',
            r'(swollen|swelling|tender|stiff|numb|tingling)',
            r'(bruising|inflammation|discomfort)',
            r'(twisted|sprained|injured|hurt|damaged)\s+\w+'
        ]
        
        symptoms = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    symptoms.append(' '.join(match))
                else:
                    symptoms.append(match)
        
        return symptoms
    
    def _extract_treatment_patterns(self, text: str) -> List[str]:
        """Extract treatments using pattern matching."""
        patterns = [
            r'(taking|prescribed|given|using)\s+(\w+)',
            r'(rest|ice|elevation|crutches|brace|physiotherapy|therapy)',
            r'(ibuprofen|medication|surgery|exercise)',
            r'(\d+\s+weeks?|months?)\s+of\s+(\w+)'
        ]
        
        treatments = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    treatments.append(' '.join(match))
                else:
                    treatments.append(match)
        
        return treatments
    
    def _extract_diagnosis_patterns(self, text: str) -> List[str]:
        """Extract diagnoses using pattern matching."""
        patterns = [
            r'(grade\s+\d+\s+sprain)',
            r'(fractured?|broken)\s+\w+',
            r'(diagnosed with|suffering from)\s+(\w+)',
            r'(acute|chronic|severe|mild)\s+(\w+)'
        ]
        
        diagnoses = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    diagnoses.append(' '.join(match))
                else:
                    diagnoses.append(match)
        
        return diagnoses
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important medical keywords and phrases."""
        # Tokenize and clean
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # POS tagging to find nouns and adjectives
        pos_tags = pos_tag(words)
        keywords = []
        
        for word, pos in pos_tags:
            if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
                if len(word) > 2:  # Filter out very short words
                    keywords.append(word)
        
        # Extract multi-word medical phrases
        medical_phrases = self._extract_medical_phrases(text)
        keywords.extend(medical_phrases)
        
        return list(set(keywords))
    
    def _extract_medical_phrases(self, text: str) -> List[str]:
        """Extract common medical phrases."""
        phrases = [
            r'grade\s+\d+\s+sprain',
            r'physical\s+therapy',
            r'ankle\s+brace',
            r'range\s+of\s+motion',
            r'recovery\s+time',
            r'weight\s+bearing',
            r'sports\s+injury',
            r'follow\s+up',
            r'pain\s+management'
        ]
        
        extracted_phrases = []
        for pattern in phrases:
            matches = re.findall(pattern, text.lower())
            extracted_phrases.extend(matches)
        
        return extracted_phrases
    
    def analyze_sentiment(self, statement: str) -> str:
        """Analyze sentiment of patient statements."""
        statement_lower = statement.lower()
        
        # Count anxiety and reassurance keywords
        anxiety_count = sum(1 for word in self.anxiety_keywords if word in statement_lower)
        reassurance_count = sum(1 for word in self.reassurance_keywords if word in statement_lower)
        
        # Pattern-based sentiment analysis
        if any(phrase in statement_lower for phrase in ['worried', 'concerned', 'scared', 'afraid']):
            return 'Anxious'
        elif any(phrase in statement_lower for phrase in ['good', 'better', 'glad', 'thankful', 'relief']):
            return 'Reassured'
        elif anxiety_count > reassurance_count:
            return 'Anxious'
        elif reassurance_count > anxiety_count:
            return 'Reassured'
        else:
            return 'Neutral'
    
    def detect_intent(self, statement: str) -> str:
        """Detect intent of patient statements."""
        statement_lower = statement.lower()
        
        # Intent patterns
        if any(phrase in statement_lower for phrase in ['hurt', 'pain', 'sore', 'ache', 'swollen']):
            return 'Reporting symptoms'
        elif any(phrase in statement_lower for phrase in ['when can', 'how long', 'will i', 'can i']):
            return 'Seeking reassurance'
        elif any(phrase in statement_lower for phrase in ['worried', 'concerned', 'what if']):
            return 'Expressing concern'
        elif any(phrase in statement_lower for phrase in ['taking', 'using', 'doing', 'been']):
            return 'Describing treatment'
        elif any(phrase in statement_lower for phrase in ['what', 'how', 'when', 'where']):
            return 'Clarifying recovery'
        else:
            return 'General discussion'
    
    def generate_medical_summary(self, transcript: str) -> Dict[str, Any]:
        """Generate structured medical summary from transcript."""
        parsed = self.parse_transcript(transcript)
        full_text = ' '.join(parsed['physician'] + parsed['patient'])
        
        # Extract medical entities
        entities = self.extract_medical_entities(full_text)
        
        # Extract patient name (simplified - looking for common patterns)
        patient_name = self._extract_patient_name(transcript)
        
        # Generate summary
        summary = {
            "Patient_Name": patient_name,
            "Symptoms": entities['symptoms'],
            "Diagnosis": entities['diagnoses'],
            "Treatment": entities['treatments'],
            "Current_Status": self._extract_current_status(full_text),
            "Prognosis": self._extract_prognosis(full_text),
            "Key_Phrases": self.extract_keywords(full_text)
        }
        
        return summary
    
    def _extract_patient_name(self, transcript: str) -> str:
        """Extract patient name from transcript."""
        # Look for common patterns
        patterns = [
            r'Mr\.\s+(\w+)',
            r'Mrs\.\s+(\w+)',
            r'Ms\.\s+(\w+)',
            r'Patient:\s+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, transcript)
            if match:
                return match.group(1)
        
        return "Unknown"
    
    def _extract_current_status(self, text: str) -> str:
        """Extract current status from text."""
        status_patterns = [
            r'healing\s+well',
            r'recovering\s+well',
            r'still\s+(\w+)',
            r'currently\s+(\w+)',
            r'now\s+(\w+)'
        ]
        
        for pattern in status_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return "Under treatment"
    
    def _extract_prognosis(self, text: str) -> str:
        """Extract prognosis from text."""
        prognosis_patterns = [
            r'recover\s+fully\s+in\s+(\d+[–-]\d+\s+weeks?)',
            r'should\s+be\s+(\w+)\s+in\s+(\d+\s+weeks?)',
            r'full\s+recovery\s+in\s+(\d+\s+weeks?)',
            r'(\d+[–-]\d+\s+weeks?)\s+for\s+full\s+recovery'
        ]
        
        for pattern in prognosis_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return "Good with proper treatment"
    
    def analyze_patient_sentiment_intent(self, transcript: str) -> List[Dict[str, str]]:
        """Analyze sentiment and intent for each patient statement."""
        parsed = self.parse_transcript(transcript)
        results = []
        
        for statement in parsed['patient']:
            if statement.strip():
                sentiment = self.analyze_sentiment(statement)
                intent = self.detect_intent(statement)
                
                results.append({
                    "Patient_Statement": statement,
                    "Sentiment": sentiment,
                    "Intent": intent
                })
        
        return results
    
    def generate_soap_note(self, transcript: str) -> Dict[str, Any]:
        """Generate SOAP note from transcript."""
        parsed = self.parse_transcript(transcript)
        full_text = ' '.join(parsed['physician'] + parsed['patient'])
        
        # Extract SOAP components
        soap_note = {
            "Subjective": {
                "Chief_Complaint": self._extract_chief_complaint(parsed['patient']),
                "History_of_Present_Illness": self._extract_history(parsed['patient'])
            },
            "Objective": {
                "Physical_Exam": self._extract_physical_exam(parsed['physician']),
                "Observations": self._extract_observations(parsed['physician'])
            },
            "Assessment": {
                "Diagnosis": self._extract_diagnosis_soap(full_text),
                "Severity": self._extract_severity(full_text)
            },
            "Plan": {
                "Treatment": self._extract_treatment_plan(parsed['physician']),
                "Follow_Up": self._extract_follow_up(parsed['physician'])
            }
        }
        
        return soap_note
    
    def _extract_chief_complaint(self, patient_statements: List[str]) -> str:
        """Extract chief complaint from patient statements."""
        if patient_statements:
            return patient_statements[0] if patient_statements[0] else "Not specified"
        return "Not specified"
    
    def _extract_history(self, patient_statements: List[str]) -> str:
        """Extract history of present illness."""
        history_statements = []
        for statement in patient_statements:
            if any(word in statement.lower() for word in ['last', 'yesterday', 'ago', 'when', 'happened']):
                history_statements.append(statement)
        return '. '.join(history_statements[:3]) if history_statements else "Not detailed"
    
    def _extract_physical_exam(self, physician_statements: List[str]) -> str:
        """Extract physical exam findings."""
        exam_statements = []
        for statement in physician_statements:
            if any(word in statement.lower() for word in ['exam', 'check', 'look', 'feel', 'range']):
                exam_statements.append(statement)
        return '. '.join(exam_statements) if exam_statements else "Physical examination performed"
    
    def _extract_observations(self, physician_statements: List[str]) -> str:
        """Extract clinical observations."""
        obs_statements = []
        for statement in physician_statements:
            if any(word in statement.lower() for word in ['healing', 'swelling', 'stable', 'intact']):
                obs_statements.append(statement)
        return '. '.join(obs_statements) if obs_statements else "Clinical observations documented"
    
    def _extract_diagnosis_soap(self, text: str) -> str:
        """Extract diagnosis for SOAP note."""
        diagnosis_patterns = [
            r'grade\s+\d+\s+sprain',
            r'diagnosed\s+with\s+(\w+)',
            r'it\s+was\s+a\s+(\w+)'
        ]
        
        for pattern in diagnosis_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(0)
        
        return "Clinical diagnosis pending"
    
    def _extract_severity(self, text: str) -> str:
        """Extract severity assessment."""
        if 'grade 2' in text.lower():
            return 'Moderate'
        elif 'mild' in text.lower():
            return 'Mild'
        elif 'severe' in text.lower():
            return 'Severe'
        else:
            return 'Moderate'
    
    def _extract_treatment_plan(self, physician_statements: List[str]) -> str:
        """Extract treatment plan."""
        treatment_statements = []
        for statement in physician_statements:
            if any(word in statement.lower() for word in ['rest', 'therapy', 'avoid', 'start', 'continue']):
                treatment_statements.append(statement)
        return '. '.join(treatment_statements) if treatment_statements else "Treatment plan documented"
    
    def _extract_follow_up(self, physician_statements: List[str]) -> str:
        """Extract follow-up plan."""
        followup_statements = []
        for statement in physician_statements:
            if any(word in statement.lower() for word in ['follow', 'return', 'come back', 'reassess']):
                followup_statements.append(statement)
        return '. '.join(followup_statements) if followup_statements else "Follow-up as needed"
    
    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """Process the complete transcript and return all outputs."""
        return {
            "medical_summary": self.generate_medical_summary(transcript),
            "sentiment_intent_analysis": self.analyze_patient_sentiment_intent(transcript),
            "soap_note": self.generate_soap_note(transcript)
        }


def main():
    """Main function to demonstrate the Physician Notetaker pipeline."""
    
    # Sample transcript
    sample_transcript = """
Physician: Good morning, Mr. Patel. What brings you in today?

Patient: Good morning, doctor. I twisted my ankle pretty badly last weekend while playing football, and it's still swollen and sore.

Physician: I see. Can you tell me exactly what happened?

Patient: It was last Saturday, around 4 PM. I was playing a friendly match at our local sports ground. I went to make a quick turn, but my foot caught the edge of the turf and rolled inward. I felt a sharp pain right away and had to stop playing.

Physician: That sounds painful. Were you able to walk after that?

Patient: Barely. I limped off the field and iced it as soon as I got home, but it swelled up a lot that evening.

Physician: Did you go to urgent care or get it checked that day?

Patient: Yes, I went to the urgent care center the next morning. They took an X-ray, and luckily there was no fracture. They said it was a grade 2 sprain.

Physician: That's good news. Have you been keeping weight off it since then?

Patient: Mostly, yes. I've been using crutches and elevating it when I can. I also wear an ankle brace during the day for support.

Physician: Have you taken anything for the pain?

Patient: Just ibuprofen for the swelling and pain. It's manageable now but still sore, especially in the morning or after walking for a while.

Physician: And how has it affected your day-to-day life?

Patient: I've taken a few days off work. I work from home, so it's not too bad, but I can't walk around much. I also had to cancel a weekend hiking trip.

Physician: That's understandable. Any numbness, tingling, or bruising?

Patient: There was some bruising around the outside of the ankle, but it's fading. No numbness or tingling.

Physician: Great. Let me do a quick physical exam to check your range of motion and stability.

[Physical Examination Conducted]

Physician: Your ankle is healing well. There's some stiffness and mild swelling, but your ligaments seem intact and stable. With rest and physiotherapy, you should recover fully in about 4–6 weeks.

Patient: That's good to know. When can I start walking without crutches?

Physician: You can gradually start putting more weight on it over the next week. Use the brace for support, and start gentle ankle movements to avoid stiffness. Avoid high-impact activities for now.

Patient: Got it. Can I resume sports eventually?

Physician: Yes, but ease into it. Once you have full range of motion and no pain, you can start light jogging. Full sports should wait until you're pain-free and confident in your stability—probably in 6–8 weeks.

Patient: Makes sense. Thanks for the guidance.

Physician: You're welcome, Mr. Patel. You're on the right track. If anything worsens or doesn't improve, come back in two weeks for a reassessment.

Patient: Will do. Thanks again, doctor.

Physician: Take care, and good luck with your recovery!
"""
    
    # Initialize the notetaker
    notetaker = PhysicianNotetaker()
    
    # Process the transcript
    results = notetaker.process_transcript(sample_transcript)
    
    # Print results
    print("=" * 80)
    print("PHYSICIAN NOTETAKER NLP PIPELINE RESULTS")
    print("=" * 80)
    
    print("\n1. MEDICAL SUMMARY:")
    print(json.dumps(results["medical_summary"], indent=2))
    
    print("\n2. SENTIMENT & INTENT ANALYSIS:")
    print(json.dumps(results["sentiment_intent_analysis"], indent=2))
    
    print("\n3. SOAP NOTE:")
    print(json.dumps(results["soap_note"], indent=2))
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()