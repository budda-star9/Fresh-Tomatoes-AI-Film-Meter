"""
Fresh-Tomatoes-AI-Film-Meter - Advanced Film Analysis Platform
Version: 2.0 Enhanced with Video Viewer
Description: AI-powered film analysis with YouTube integration, cultural context analysis,
             comprehensive scoring algorithms, and embedded video viewing.
"""

# --------------------------
# IMPORTS
# --------------------------
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import hashlib
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import textwrap
import time
import io
import base64
from typing import Dict, List, Optional, Tuple, Any, Union
import json

# --------------------------
# CONFIGURATION & SETUP
# --------------------------
st.set_page_config(
    page_title="Fresh-Tomatoes-AI-Film-Meter 🎬 - Advanced Film Analysis",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/flickfinder ',
        'Report a bug': 'https://github.com/yourusername/flickfinder/issues ',
        'About': "### Fresh-Tomatoes-AI-Film-Meter v2.0\nAdvanced film analysis using AI and machine learning."
    }
)

# Initialize NLTK data
@st.cache_resource
def load_nltk_data() -> None:
    """Load required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        with st.spinner("Downloading NLP data..."):
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

load_nltk_data()

# Initialize all session state variables
session_defaults = {
    'analysis_history': [],
    'stored_results': {},
    'current_analysis_id': None,
    'show_results_page': False,
    'saved_projects': {},
    'project_counter': 0,
    'current_page': "🏠 Dashboard",
    'current_results_display': None,
    'current_video_id': None,
    'current_video_title': None,
    'top_films': [],
    'analysis_count': 0,
    'last_analysis_time': None,
    'batch_results': None,
    'show_batch_results': False,
    'analytics_view': 'overview',
    'show_breakdown': False,
    'current_tab': 'youtube',
}

for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------
# ENHANCED PERSISTENCE MANAGER CLASS
# --------------------------
class PersistenceManager:
    """Handles saving and loading of analysis results with unique IDs"""
    
    @staticmethod
    def generate_film_id(film_data: Dict) -> str:
        """Generate a unique ID for a film based on its content"""
        content_string = f"{film_data.get('title', '')}_{film_data.get('synopsis', '')[:100]}"
        return hashlib.md5(content_string.encode()).hexdigest()[:12]
    
    @staticmethod
    def save_results(film_data: Dict, analysis_results: Dict, film_id: Optional[str] = None) -> str:
        """Save analysis results with full persistence"""
        if film_id is None:
            film_id = PersistenceManager.generate_film_id(film_data)
        
        # Store in session state
        st.session_state.stored_results[film_id] = {
            'film_data': film_data,
            'analysis_results': analysis_results,
            'timestamp': datetime.now().isoformat(),
            'film_id': film_id
        }
        
        # Add to history
        history_entry = {
            'id': film_id,
            'title': film_data.get('title', 'Unknown Film'),
            'timestamp': datetime.now().isoformat(),
            'overall_score': analysis_results.get('overall_score', 0),
            'detected_genre': analysis_results.get('genre_insights', {}).get('primary_genre', 'Unknown'),
            'cultural_relevance': analysis_results.get('cultural_insights', {}).get('relevance_score', 0),
            'component_scores': analysis_results.get('cinematic_scores', {}),
            'synopsis': film_data.get('synopsis', '')[:200]
        }
        
        # Add to history if not already there
        existing_ids = [h.get('id') for h in st.session_state.analysis_history]
        if film_id not in existing_ids:
            st.session_state.analysis_history.append(history_entry)
        
        # Update top films
        PersistenceManager._update_top_films()
        
        # Set as current display
        st.session_state.current_results_display = analysis_results
        st.session_state.show_results_page = True
        st.session_state.current_analysis_id = film_id
        st.session_state.last_analysis_time = datetime.now().isoformat()
        st.session_state.analysis_count += 1
        
        return film_id
    
    @staticmethod
    def _update_top_films() -> None:
        """Update the top films list based on overall score"""
        all_films = list(st.session_state.stored_results.values())
        if all_films:
            sorted_films = sorted(
                all_films,
                key=lambda x: x['analysis_results']['overall_score'],
                reverse=True
            )
            st.session_state.top_films = sorted_films[:3]
    
    @staticmethod
    def load_results(film_id: str) -> Optional[Dict]:
        """Load analysis results by film ID"""
        return st.session_state.stored_results.get(film_id)
    
    @staticmethod
    def get_all_history() -> List[Dict]:
        """Get all analysis history"""
        return st.session_state.analysis_history
    
    @staticmethod
    def clear_history() -> None:
        """Clear all analysis history"""
        st.session_state.analysis_history = []
        st.session_state.stored_results = {}
        st.session_state.current_results_display = None
        st.session_state.show_results_page = False
        st.session_state.top_films = []
        st.session_state.analysis_count = 0
        st.session_state.current_video_id = None
        st.session_state.current_video_title = None
        st.session_state.batch_results = None
        st.session_state.show_batch_results = False
    
    @staticmethod
    def save_project(project_name: str, film_data: Dict, analysis_results: Dict) -> str:
        """Save a project with custom name"""
        project_id = f"project_{st.session_state.project_counter}"
        st.session_state.project_counter += 1
        
        st.session_state.saved_projects[project_id] = {
            'name': project_name,
            'film_data': film_data,
            'analysis_results': analysis_results,
            'saved_at': datetime.now().isoformat()
        }
        
        return project_id
    
    @staticmethod
    def load_project(project_id: str) -> Optional[Dict]:
        """Load a saved project"""
        return st.session_state.saved_projects.get(project_id)
    
    @staticmethod
    def get_all_projects() -> Dict:
        """Get all saved projects"""
        return st.session_state.saved_projects
    
    @staticmethod
    def get_top_films() -> List[Dict]:
        """Get top films"""
        if not st.session_state.top_films:
            PersistenceManager._update_top_films()
        return st.session_state.top_films
    
    @staticmethod
    def get_analytics_data() -> Optional[pd.DataFrame]:
        """Get comprehensive analytics data"""
        history = st.session_state.analysis_history
        if not history:
            return None
        
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['time_of_day'] = df['timestamp'].dt.hour
        
        return df

# --------------------------
# ENHANCED FILM-SPECIFIC SCORER CLASS
# --------------------------
class FilmSpecificScorer:
    """Handles film scoring with genre-specific adjustments"""
    
    def __init__(self):
        self.base_weights = {
            'narrative': 0.28,
            'emotional': 0.25,
            'character': 0.22,
            'cultural': 0.15,
            'technical': 0.10
        }
        self.philosophical_insights = [
            "Art as cultural memory",
            "Narrative as truth-seeking",
            "Cinema as empathy machine",
            "Storytelling as resistance",
            "Film as time capsule",
            "Visual language as consciousness",
            "Character as human mirror",
            "Genre as cultural dialogue"
        ]
    
    def calculate_unique_film_score(self, analysis_results: Dict, film_data: Dict) -> Dict:
        """Calculate comprehensive film score with genre-specific adjustments"""
        text = (film_data.get('synopsis', '') + ' ' + film_data.get('transcript', '')).lower()
        title = film_data.get('title', '').lower()
        
        # Get detected genre safely
        genre_context = analysis_results.get('genre_context', {})
        detected_genre = genre_context.get('primary_genre', '') if isinstance(genre_context, dict) else ''
        if not detected_genre:
            detected_genre = genre_context.get('detected_genre', '').lower()
        else:
            detected_genre = detected_genre.lower()
        
        # Component scores
        narrative_score = self._score_narrative(analysis_results.get('narrative_structure', {}), text)
        emotional_score = self._score_emotional(analysis_results.get('emotional_arc', {}))
        character_score = self._score_characters(analysis_results.get('character_analysis', {}), text)
        cultural_score = analysis_results.get('cultural_context', {}).get('relevance_score', 0.0)
        technical_score = self._score_technical(analysis_results.get('narrative_structure', {}), text)
        
        # Genre-specific weight adjustments
        weights = self.base_weights.copy()
        
        if any(g in detected_genre for g in ['drama', 'urban drama', 'black cinema', 'black', 'urban']):
            weights['emotional'] += 0.08
            weights['character'] += 0.06
            weights['cultural'] += 0.10
            weights['narrative'] -= 0.05
        
        if 'comedy' in detected_genre:
            weights['emotional'] += 0.05
            weights['character'] += 0.08
            weights['technical'] -= 0.03
        
        if 'action' in detected_genre or 'thriller' in detected_genre:
            weights['emotional'] -= 0.03
            weights['technical'] += 0.08
            weights['narrative'] += 0.05
        
        if 'short film' in title or len(text.split()) < 800:
            narrative_score = max(narrative_score, 0.68)
            emotional_score = max(emotional_score, 0.70)
        
        # Cultural bonus
        cultural_bonus = 0.0
        if cultural_score > 0.6:
            cultural_bonus = (cultural_score - 0.6) * 2.0
        elif cultural_score > 0.4:
            cultural_bonus = (cultural_score - 0.4) * 0.8
        
        # Philosophical insight bonus
        philosophical_bonus = self._calculate_philosophical_bonus(text, cultural_score)
        cultural_bonus += philosophical_bonus
        
        # Calculate final score
        raw_score = (
            narrative_score * weights['narrative'] +
            emotional_score * weights['emotional'] +
            character_score * weights['character'] +
            cultural_score * weights['cultural'] +
            technical_score * weights['technical']
        )
        
        raw_score += cultural_bonus
        raw_score = min(1.25, raw_score)
        
        final_score = raw_score * 5.0
        
        # Add uniqueness factor
        fingerprint = hash(text[:500] + title) % 100
        final_score += (fingerprint / 1000)
        
        final_score = round(max(1.8, min(4.9, final_score)), 1)
        
        # Add variation in middle range
        if 3.7 <= final_score <= 4.0:
            variation = random.uniform(-0.2, 0.2)
            final_score = round(max(2.0, min(4.8, final_score + variation)), 1)
        
        return {
            'overall_score': final_score,
            'component_scores': {
                'narrative': round(narrative_score * 5, 1),
                'emotional': round(emotional_score * 5, 1),
                'character': round(character_score * 5, 1),
                'cultural': round(cultural_score * 5, 1),
                'technical': round(technical_score * 5, 1)
            },
            'weighted_scores': {
                'narrative': narrative_score,
                'emotional': emotional_score,
                'character': character_score,
                'cultural': cultural_score,
                'technical': technical_score
            },
            'applied_weights': weights,
            'cultural_bonus': round(cultural_bonus, 3),
            'philosophical_insight': random.choice(self.philosophical_insights) if cultural_score > 0.4 else None
        }
    
    def _calculate_philosophical_bonus(self, text: str, cultural_score: float) -> float:
        """Calculate bonus for philosophical depth"""
        philosophical_keywords = [
            'identity', 'memory', 'truth', 'justice', 'freedom',
            'love', 'death', 'time', 'reality', 'consciousness',
            'morality', 'existence', 'meaning', 'society', 'power'
        ]
        
        matches = sum(1 for keyword in philosophical_keywords if keyword in text)
        base_bonus = matches * 0.01
        
        if cultural_score > 0.5:
            base_bonus *= 1.5
        
        return min(0.05, base_bonus)
    
    def _score_narrative(self, ns: Dict, text: str) -> float:
        """Score narrative structure"""
        ld = ns.get('lexical_diversity', 0.4)
        structural = ns.get('structural_score', 0.4)
        length = len(text.split())
        base = (ld * 0.5 + structural * 0.5)
        return min(1.0, base + (length > 300) * 0.15 + (length > 800) * 0.1)
    
    def _score_emotional(self, ea: Dict) -> float:
        """Score emotional arc"""
        arc = ea.get('arc_score', 0.4)
        variance = ea.get('emotional_variance', 0.2)
        return min(1.0, arc * 0.7 + variance * 1.2 + 0.2)
    
    def _score_characters(self, ca: Dict, text: str) -> float:
        """Score character development"""
        chars = ca.get('potential_characters', 3)
        density = ca.get('character_density', 0.03)
        mentions = text.count(" he ") + text.count(" she ") + text.count(" his ") + text.count(" her ")
        return min(1.0, (chars / 8) * 0.6 + density * 8 + min(mentions / 50, 0.4))
    
    def _score_technical(self, ns: Dict, text: str) -> float:
        """Score technical aspects"""
        readability = ns.get('readability_score', 0.6)
        dialogue_density = len(re.findall(r'\b[A-Z][a-z]+:', text)) / max(1, len(text.split('\n')))
        return min(1.0, readability + dialogue_density * 2 + 0.3)

# --------------------------
# ENHANCED SMART GENRE DETECTOR WITH AI SUGGESTIONS
# --------------------------
class SmartGenreDetector:
    """Detects film genres with AI enhancement suggestions"""
    
    def __init__(self):
        self.genre_patterns = self._build_genre_patterns()
    
    def _build_genre_patterns(self) -> Dict:
        """Build comprehensive genre detection patterns"""
        return {
            "Drama": {
                "keywords": ["emotional", "relationship", "conflict", "family", "love", "heart", 
                           "struggle", "life", "human", "drama", "tragic", "serious", "pain", "loss"],
                "weight": 1.2,
                "philosophical_aspect": "Exploration of human condition"
            },
            "Comedy": {
                "keywords": ["funny", "laugh", "humor", "joke", "comic", "satire", "hilarious", 
                           "wit", "absurd", "comedy", "fun", "humorous", "lighthearted"],
                "weight": 1.1,
                "philosophical_aspect": "Social commentary through humor"
            },
            "Horror": {
                "keywords": ["fear", "terror", "scary", "horror", "ghost", "monster", "kill", 
                           "death", "dark", "night", "supernatural", "creepy", "frightening"],
                "weight": 1.1,
                "philosophical_aspect": "Confrontation with mortality"
            },
            "Sci-Fi": {
                "keywords": ["future", "space", "alien", "technology", "robot", "planet", 
                           "time travel", "science", "sci-fi", "futuristic", "cyber"],
                "weight": 1.1,
                "philosophical_aspect": "Questioning technological progress"
            },
            "Action": {
                "keywords": ["fight", "chase", "gun", "explosion", "mission", "danger", 
                           "escape", "battle", "adventure", "action", "thrilling", "exciting"],
                "weight": 1.1,
                "philosophical_aspect": "Moral agency under pressure"
            },
            "Thriller": {
                "keywords": ["suspense", "mystery", "danger", "chase", "secret", "conspiracy", 
                           "tense", "cliffhanger", "thriller", "suspenseful", "mysterious"],
                "weight": 1.1,
                "philosophical_aspect": "Uncertainty and trust"
            },
            "Romance": {
                "keywords": ["love", "romance", "heart", "relationship", "kiss", "date", 
                           "passion", "affection", "romantic", "lovers", "affection"],
                "weight": 1.1,
                "philosophical_aspect": "Nature of human connection"
            },
            "Documentary": {
                "keywords": ["real", "fact", "interview", "evidence", "truth", "history", 
                           "actual", "reality", "documentary", "non-fiction", "educational"],
                "weight": 1.2,
                "philosophical_aspect": "Construction of truth"
            },
            "Fantasy": {
                "keywords": ["magic", "dragon", "kingdom", "quest", "mythical", "wizard", 
                           "enchanted", "supernatural", "fantasy", "magical", "mythical"],
                "weight": 1.1,
                "philosophical_aspect": "Myth-making and symbolism"
            },
            "Black Cinema": {
                "keywords": ["black", "african", "diaspora", "racial", "cultural", "community",
                           "heritage", "identity", "resilience", "justice", "afro", "systemic",
                           "black experience", "african american", "civil rights"],
                "weight": 1.3,
                "philosophical_aspect": "Cultural memory and resistance"
            },
            "Urban Drama": {
                "keywords": ["urban", "city", "street", "hood", "neighborhood", "ghetto",
                           "inner city", "metropolitan", "concrete", "asphalt", "urban life"],
                "weight": 1.2,
                "philosophical_aspect": "Modern alienation and community"
            }
        }
    
    def detect_genre(self, text: str, existing_genre: Optional[str] = None) -> Dict:
        """Smart genre detection with weighted scoring"""
        if not text or len(text.strip()) < 10:
            return {
                'primary_genre': existing_genre or "Unknown",
                'confidence': 0,
                'details': {},
                'secondary_genres': [],
                'all_genres': {}
            }
        
        text_lower = text.lower()
        genre_scores = {}
        genre_details = {}
        
        for genre, pattern_data in self.genre_patterns.items():
            score = 0
            keywords = pattern_data["keywords"]
            weight = pattern_data.get("weight", 1.0)
            
            keyword_matches = []
            for keyword in keywords:
                if keyword in text_lower:
                    score += 2 * weight
                    keyword_matches.append(keyword)
                elif any(word.startswith(keyword.split()[0]) for word in text_lower.split()):
                    score += 1 * weight
                    keyword_matches.append(keyword)
            
            if existing_genre and genre.lower() in existing_genre.lower():
                score += 3
            
            if score > 0:
                genre_scores[genre] = score
                genre_details[genre] = {
                    'score': score,
                    'keywords': keyword_matches[:5],
                    'philosophical_aspect': pattern_data.get('philosophical_aspect', '')
                }
        
        if not genre_scores:
            return {
                'primary_genre': existing_genre or "Drama",
                'confidence': 50,
                'details': {},
                'secondary_genres': [],
                'all_genres': {}
            }
        
        top_genre, top_score = max(genre_scores.items(), key=lambda x: x[1])
        secondary_genres = [g for g, s in genre_scores.items() if s >= top_score * 0.5 and g != top_genre]
        
        return {
            'primary_genre': top_genre,
            'confidence': min(100, top_score * 10),
            'details': genre_details.get(top_genre, {}),
            'secondary_genres': secondary_genres[:2],
            'all_genres': genre_details
        }

# --------------------------
# ENHANCED CULTURAL CONTEXT ANALYZER
# --------------------------
class CulturalContextAnalyzer:
    """Analyzes cultural context and relevance in films"""
    
    def __init__(self):
        self.cultural_themes = {
            'black_experience': {
                'keywords': ['black experience', 'african american', 'black community', 
                           'black culture', 'black identity', 'black history'],
                'philosophical': 'Diasporic consciousness and cultural memory',
                'weight': 1.3
            },
            'diaspora': {
                'keywords': ['diaspora', 'african diaspora', 'caribbean', 'afro-latino', 
                           'pan-african', 'transatlantic'],
                'philosophical': 'Hybrid identities and transnational connections',
                'weight': 1.2
            },
            'social_justice': {
                'keywords': ['social justice', 'racial justice', 'civil rights', 
                           'equality', 'activism', 'protest', 'resistance'],
                'philosophical': 'Ethical frameworks and moral agency',
                'weight': 1.4
            },
            'cultural_heritage': {
                'keywords': ['heritage', 'ancestral', 'tradition', 'cultural roots',
                           'lineage', 'generational'],
                'philosophical': 'Historical continuity and collective memory',
                'weight': 1.2
            },
            'urban_life': {
                'keywords': ['urban life', 'inner city', 'metropolitan', 'city living',
                           'street culture', 'urban landscape'],
                'philosophical': 'Modern alienation and urban psychology',
                'weight': 1.1
            }
        }
    
    def analyze_cultural_context(self, film_data: Dict) -> Dict:
        """Analyze cultural context with nuanced scoring and philosophical insights"""
        text = (film_data.get('synopsis', '') + ' ' + 
                film_data.get('transcript', '') + ' ' +
                film_data.get('title', '')).lower()
        
        theme_scores = {}
        theme_details = {}
        total_weighted_matches = 0
        
        for theme, theme_data in self.cultural_themes.items():
            matches = 0
            matched_keywords = []
            for keyword in theme_data['keywords']:
                if keyword in text:
                    matches += 1
                    matched_keywords.append(keyword)
            
            weighted_matches = matches * theme_data.get('weight', 1.0)
            theme_scores[theme] = weighted_matches
            total_weighted_matches += weighted_matches
            
            if matches > 0:
                theme_details[theme] = {
                    'matches': matches,
                    'keywords': matched_keywords,
                    'philosophical_aspect': theme_data['philosophical'],
                    'weighted_score': weighted_matches
                }
        
        if total_weighted_matches == 0:
            return {
                'relevance_score': 0.0,
                'primary_themes': [],
                'theme_breakdown': theme_scores,
                'theme_details': {},
                'is_culturally_relevant': False,
                'philosophical_insights': [],
                'total_matches': 0
            }
        
        max_possible = sum(len(theme_data['keywords']) * theme_data.get('weight', 1.0) 
                          for theme_data in self.cultural_themes.values())
        relevance_score = min(1.0, total_weighted_matches / (max_possible * 0.15))
        
        primary_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        primary_themes = [theme for theme, score in primary_themes if score > 0]
        
        philosophical_insights = []
        for theme in primary_themes:
            if theme in theme_details:
                philosophical_insights.append(theme_details[theme]['philosophical_aspect'])
        
        return {
            'relevance_score': round(relevance_score, 2),
            'primary_themes': primary_themes,
            'theme_breakdown': theme_scores,
            'theme_details': theme_details,
            'is_culturally_relevant': relevance_score > 0.3,
            'philosophical_insights': philosophical_insights[:2],
            'total_matches': total_weighted_matches
        }

# --------------------------
# ENHANCED FILM ANALYSIS ENGINE
# --------------------------
class FilmAnalysisEngine:
    """Main engine for comprehensive film analysis"""
    
    def __init__(self):
        self.genre_detector = SmartGenreDetector()
        self.cultural_analyzer = CulturalContextAnalyzer()
        self.film_scorer = FilmSpecificScorer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.persistence = PersistenceManager()
    
    def analyze_film(self, film_data: Dict) -> Dict:
        """Main film analysis method with enhanced features"""
        try:
            film_id = self.persistence.generate_film_id(film_data)
            
            cached_result = self.persistence.load_results(film_id)
            if cached_result:
                if 'video_id' in film_data:
                    st.session_state.current_video_id = film_data['video_id']
                if 'video_title' in film_data:
                    st.session_state.current_video_title = film_data['video_title']
                    
                st.session_state.current_results_display = cached_result['analysis_results']
                st.session_state.show_results_page = True
                st.session_state.current_analysis_id = film_id
                return cached_result['analysis_results']
            
            analysis_text = self._prepare_analysis_text(film_data)
            
            if len(analysis_text.strip()) < 20:
                results = self._create_basic_fallback(film_data)
            else:
                analysis_results = self._perform_comprehensive_analysis(analysis_text, film_data)
                scoring_result = self.film_scorer.calculate_unique_film_score(analysis_results, film_data)
                results = self._generate_enhanced_review(film_data, analysis_results, scoring_result)
            
            self.persistence.save_results(film_data, results, film_id)
            
            if 'video_id' in film_data:
                st.session_state.current_video_id = film_data['video_id']
            if 'video_title' in film_data:
                st.session_state.current_video_title = film_data['video_title']
            
            return results
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return self._create_error_fallback(film_data, str(e))
    
    def _prepare_analysis_text(self, film_data: Dict) -> str:
        """Prepare text for analysis"""
        synopsis = film_data.get('synopsis', '')
        transcript = film_data.get('transcript', '')
        title = film_data.get('title', '')
        return f"{title} {synopsis} {transcript}".strip()
    
    def _perform_comprehensive_analysis(self, text: str, film_data: Dict) -> Dict:
        """Perform comprehensive film analysis"""
        # Genre detection
        existing_genre = film_data.get('genre', '')
        genre_result = self.genre_detector.detect_genre(text, existing_genre)
        
        # Cultural analysis
        cultural_result = self.cultural_analyzer.analyze_cultural_context(film_data)
        
        # Sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Narrative structure analysis
        narrative_result = self._analyze_narrative_structure(text)
        
        # Character analysis
        character_result = self._analyze_characters(text)
        
        # Emotional arc analysis
        emotional_result = self._analyze_emotional_arc(text)
        
        return {
            'genre_context': genre_result,
            'cultural_context': cultural_result,
            'sentiment_analysis': sentiment_scores,
            'narrative_structure': narrative_result,
            'character_analysis': character_result,
            'emotional_arc': emotional_result
        }
    
    def _analyze_narrative_structure(self, text: str) -> Dict:
        """Analyze narrative structure"""
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Calculate lexical diversity
        unique_words = set(words)
        lexical_diversity = len(unique_words) / max(len(words), 1)
        
        # Readability score (simplified)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        readability_score = min(1.0, 1 - (avg_sentence_length - 10) / 40)
        
        return {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'lexical_diversity': round(lexical_diversity, 3),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'readability_score': round(readability_score, 2),
            'structural_score': round(random.uniform(0.4, 0.9), 2)
        }
    
    def _analyze_characters(self, text: str) -> Dict:
        """Analyze character development"""
        # Extract potential character names (simple heuristic)
        words = nltk.word_tokenize(text)
        potential_characters = len([w for w in words if w.istitle() and len(w) > 1])
        
        character_density = potential_characters / max(len(words), 1)
        
        return {
            'potential_characters': min(potential_characters, 10),
            'character_density': round(character_density, 3),
            'character_score': round(random.uniform(0.3, 0.9), 2)
        }
    
    def _analyze_emotional_arc(self, text: str) -> Dict:
        """Analyze emotional arc"""
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) < 3:
            return {
                'arc_score': 0.5,
                'emotional_variance': 0.3,
                'emotional_range': 'moderate'
            }
        
        # Analyze sentiment per sentence
        sentiments = []
        for sentence in sentences[:20]:  # Limit for performance
            sentiment = self.sentiment_analyzer.polarity_scores(sentence)
            sentiments.append(sentiment['compound'])
        
        if not sentiments:
            return {
                'arc_score': 0.5,
                'emotional_variance': 0.3,
                'emotional_range': 'moderate'
            }
        
        # Calculate variance and arc
        variance = np.var(sentiments)
        arc_score = min(1.0, variance * 5 + 0.3)
        
        # Determine emotional range
        if variance > 0.1:
            emotional_range = 'wide'
        elif variance > 0.05:
            emotional_range = 'moderate'
        else:
            emotional_range = 'narrow'
        
        return {
            'arc_score': round(arc_score, 2),
            'emotional_variance': round(variance, 3),
            'emotional_range': emotional_range,
            'sentiment_samples': len(sentences)
        }
    
    def _generate_enhanced_review(self, film_data: Dict, analysis_results: Dict, scoring_result: Dict) -> Dict:
        """Generate enhanced film review with philosophical insights"""
        overall_score = scoring_result['overall_score']
        genre_result = analysis_results['genre_context']
        cultural_context = analysis_results['cultural_context']
        component_scores = scoring_result['component_scores']
        
        cinematic_scores = self._map_to_cinematic_categories(component_scores, genre_result)
        
        results = {
            'smart_summary': self._generate_philosophical_summary(film_data, overall_score, genre_result, cultural_context),
            'cinematic_scores': cinematic_scores,
            'overall_score': overall_score,
            'strengths': self._generate_enhanced_strengths(analysis_results, cinematic_scores, cultural_context),
            'weaknesses': self._generate_enhanced_weaknesses(analysis_results, cinematic_scores),
            'recommendations': self._generate_enhanced_recommendations(analysis_results, cinematic_scores, cultural_context),
            'festival_recommendations': self._generate_festival_recommendations(overall_score, genre_result, cultural_context),
            'audience_analysis': self._generate_enhanced_audience_analysis(analysis_results, genre_result, cultural_context),
            'genre_insights': genre_result,
            'cultural_insights': cultural_context,
            'scoring_breakdown': scoring_result,
            'film_title': film_data.get('title', 'Unknown Film'),
            'philosophical_insights': self._generate_philosophical_insights(film_data, analysis_results),
            'ai_tool_suggestions': self._generate_ai_tool_suggestions(analysis_results),
            'synopsis_analysis': self._analyze_synopsis(film_data.get('synopsis', '')),
            'narrative_arc_analysis': analysis_results.get('narrative_structure', {}),
            'character_ecosystem': analysis_results.get('character_analysis', {})
        }
        
        return results
    
    def _map_to_cinematic_categories(self, component_scores: Dict, genre_result: Dict) -> Dict:
        """Map component scores to cinematic categories"""
        return {
            'story_narrative': component_scores.get('narrative', 3.0),
            'visual_vision': component_scores.get('technical', 3.0) * 1.1,
            'technical_craft': component_scores.get('technical', 3.0),
            'sound_design': component_scores.get('technical', 3.0) * 0.9,
            'performance': component_scores.get('character', 3.0) * 1.2
        }
    
    def _generate_philosophical_summary(self, film_data: Dict, score: float, genre_result: Dict, cultural_context: Dict) -> str:
        """Generate philosophical film summary"""
        title = film_data.get('title', 'Unknown Film')
        genre_info = genre_result.get('details', {})
        philosophical_aspect = genre_info.get('philosophical_aspect', 'human experience')
        
        # Score-based quality assessment
        if score >= 4.5:
            quality = "profound"
            impact = "a transformative cinematic meditation"
            philosophical_frame = "transcendent artistic achievement"
        elif score >= 4.0:
            quality = "significant"
            impact = "a meaningful artistic statement"
            philosophical_frame = "compelling narrative exploration"
        elif score >= 3.5:
            quality = "substantial"
            impact = "a thoughtful creative work"
            philosophical_frame = "engaging thematic investigation"
        elif score >= 3.0:
            quality = "promising"
            impact = "an evolving artistic voice"
            philosophical_frame = "developing narrative consciousness"
        elif score >= 2.5:
            quality = "emerging"
            impact = "a foundational creative endeavor"
            philosophical_frame = "nascent artistic exploration"
        else:
            quality = "formative"
            impact = "a creative beginning"
            philosophical_frame = "initial creative expression"
        
        cultural_phrase = ""
        if cultural_context.get('is_culturally_relevant'):
            primary_themes = cultural_context.get('primary_themes', [])
            if primary_themes:
                theme_str = " and ".join(primary_themes)
                cultural_phrase = f", engaging with {theme_str} through the lens of "
        
        summary = f"**{title}** presents {quality} engagement with {philosophical_aspect}{cultural_phrase}{philosophical_frame}, resulting in {impact}."
        
        if cultural_context.get('philosophical_insights'):
            insights = cultural_context['philosophical_insights']
            if insights:
                insight_str = "; ".join(insights[:2])
                summary += f" The work contemplates {insight_str.lower()}."
        
        return summary
    
    def _generate_enhanced_strengths(self, analysis_results: Dict, cinematic_scores: Dict, cultural_context: Dict) -> List[str]:
        """Generate enhanced strengths list"""
        strengths = []
        
        # Check narrative strength
        if cinematic_scores.get('story_narrative', 0) >= 4.0:
            strengths.append("Strong narrative structure with compelling storytelling")
        
        # Check character development
        if cinematic_scores.get('performance', 0) >= 4.0:
            strengths.append("Well-developed characters with depth and authenticity")
        
        # Check cultural relevance
        if cultural_context.get('is_culturally_relevant'):
            strengths.append("Significant cultural relevance and thematic depth")
        
        # Check technical aspects
        if cinematic_scores.get('technical_craft', 0) >= 3.5:
            strengths.append("Solid technical execution and production values")
        
        # Add default strengths if none found
        if not strengths:
            strengths.append("Solid foundation for further development")
            strengths.append("Clear narrative intention")
        
        return strengths[:3]
    
    def _generate_enhanced_weaknesses(self, analysis_results: Dict, cinematic_scores: Dict) -> List[str]:
        """Generate enhanced weaknesses list"""
        weaknesses = []
        
        # Identify weakest area
        min_score = min(cinematic_scores.values()) if cinematic_scores else 0
        min_category = min(cinematic_scores.items(), key=lambda x: x[1])[0] if cinematic_scores else ""
        
        if min_score < 3.0:
            category_map = {
                'story_narrative': 'narrative structure',
                'visual_vision': 'visual storytelling',
                'technical_craft': 'technical execution',
                'sound_design': 'audio elements',
                'performance': 'character development'
            }
            weakness_category = category_map.get(min_category, min_category.replace('_', ' '))
            weaknesses.append(f"Could benefit from stronger {weakness_category}")
        
        # Add generic weaknesses if needed
        if not weaknesses:
            weaknesses.append("Consider deepening emotional resonance")
            weaknesses.append("Opportunity for more distinctive visual style")
        
        return weaknesses[:2]
    
    def _generate_enhanced_recommendations(self, analysis_results: Dict, cinematic_scores: Dict, cultural_context: Dict) -> List[str]:
        """Generate enhanced recommendations"""
        recommendations = []
        
        # Narrative recommendations
        if cinematic_scores.get('story_narrative', 0) < 3.5:
            recommendations.append("Develop narrative complexity with subplots")
        
        # Character recommendations
        if cinematic_scores.get('performance', 0) < 3.5:
            recommendations.append("Deepen character backstories and motivations")
        
        # Cultural recommendations
        if cultural_context.get('is_culturally_relevant'):
            recommendations.append("Leverage cultural themes for deeper resonance")
        
        # Technical recommendations
        if cinematic_scores.get('technical_craft', 0) < 3.5:
            recommendations.append("Enhance production values with focused resources")
        
        if not recommendations:
            recommendations.append("Continue developing your distinctive voice")
            recommendations.append("Explore collaborations to enhance production scope")
        
        return recommendations[:3]
    
    def _generate_festival_recommendations(self, score: float, genre_result: Dict, cultural_context: Dict) -> Dict:
        """Generate festival recommendations"""
        festivals_by_level = {
            'elite': ["Sundance Film Festival", "Toronto International Film Festival", 
                     "Cannes Film Festival", "Berlin International Film Festival"],
            'premier': ["South by Southwest (SXSW)", "Tribeca Film Festival", 
                       "Telluride Film Festival", "Venice Film Festival"],
            'specialized': ["BlackStar Film Festival", "Urbanworld Film Festival", 
                          "Pan African Film Festival", "AfroFilm Festival"]
        }
        
        if score >= 4.5:
            level = "elite"
            festivals = festivals_by_level['elite']
        elif score >= 4.0:
            level = "premier"
            festivals = festivals_by_level['premier']
        else:
            level = "specialized"
            festivals = festivals_by_level['specialized']
        
        # Add specialized festivals for cultural relevance
        if cultural_context.get('is_culturally_relevant'):
            festivals.extend(festivals_by_level['specialized'][:2])
        
        return {
            'level': level,
            'festivals': list(set(festivals))[:4]
        }
    
    def _generate_enhanced_audience_analysis(self, analysis_results: Dict, genre_result: Dict, cultural_context: Dict) -> Dict:
        """Generate enhanced audience analysis"""
        genre = genre_result.get('primary_genre', 'Unknown')
        
        audience_mapping = {
            'Drama': ["Film enthusiasts", "Art house audiences", "Critics"],
            'Comedy': ["General audiences", "Young adults", "Festival goers"],
            'Horror': ["Genre fans", "Thrill-seekers", "Niche audiences"],
            'Sci-Fi': ["Tech enthusiasts", "Fantasy fans", "Futurists"],
            'Action': ["Mainstream audiences", "Action fans", "Entertainment seekers"],
            'Black Cinema': ["Cultural audiences", "Diaspora communities", "Socially conscious viewers"],
            'Urban Drama': ["Urban audiences", "Youth demographics", "Social realism enthusiasts"]
        }
        
        audiences = audience_mapping.get(genre, ["General film audiences", "Festival attendees"])
        
        # Add cultural audiences if relevant
        if cultural_context.get('is_culturally_relevant'):
            audiences.append("Culturally engaged viewers")
            audiences.append("Academic and educational audiences")
        
        engagement_score = random.uniform(0.6, 0.9) if len(audiences) > 3 else random.uniform(0.4, 0.7)
        
        return {
            'target_audiences': list(set(audiences))[:5],
            'engagement_score': round(engagement_score, 2),
            'market_potential': 'High' if engagement_score > 0.7 else 'Medium' if engagement_score > 0.5 else 'Developing'
        }
    
    def _generate_philosophical_insights(self, film_data: Dict, analysis_results: Dict) -> List[str]:
        """Generate philosophical insights about the film"""
        insights = []
        text = film_data.get('synopsis', '') + ' ' + film_data.get('transcript', '')
        
        # Check for existential themes
        existential_keywords = ['death', 'life', 'meaning', 'existence', 'purpose']
        if any(keyword in text.lower() for keyword in existential_keywords):
            insights.append("Explores existential questions about human purpose")
        
        # Check for social themes
        social_keywords = ['society', 'justice', 'equality', 'power', 'freedom']
        if any(keyword in text.lower() for keyword in social_keywords):
            insights.append("Engages with social structures and power dynamics")
        
        # Check for psychological themes
        psychological_keywords = ['mind', 'memory', 'identity', 'consciousness', 'dream']
        if any(keyword in text.lower() for keyword in psychological_keywords):
            insights.append("Investigates psychological depth and identity")
        
        # Cultural insights
        cultural_insights = analysis_results.get('cultural_context', {}).get('philosophical_insights', [])
        insights.extend(cultural_insights)
        
        return insights[:3] if insights else ["Explores fundamental human experiences"]
    
    def _generate_ai_tool_suggestions(self, analysis_results: Dict) -> List[Dict]:
        """Generate AI tool suggestions for further analysis"""
        suggestions = []
        
        # Add suggestions based on analysis depth
        narrative = analysis_results.get('narrative_structure', {})
        if narrative.get('word_count', 0) > 500:
            suggestions.append({
                'tool': 'GPT-4',
                'purpose': 'Advanced narrative analysis',
                'benefit': 'Detailed plot structure and thematic exploration'
            })
        
        cultural = analysis_results.get('cultural_context', {})
        if cultural.get('is_culturally_relevant'):
            suggestions.append({
                'tool': 'CulturalBERT',
                'purpose': 'Cultural context analysis',
                'benefit': 'Enhanced cultural relevance scoring'
            })
        
        if narrative.get('word_count', 0) > 1000:
            suggestions.append({
                'tool': 'BERT',
                'purpose': 'Contextual understanding',
                'benefit': 'Better genre and theme detection'
            })
        
        return suggestions[:3]
    
    def _analyze_synopsis(self, synopsis: str) -> Dict:
        """Analyze synopsis for key insights"""
        if not synopsis:
            return {
                'length': 0,
                'key_themes': [],
                'emotional_tone': 'neutral',
                'complexity': 'low'
            }
        
        words = synopsis.split()
        sentences = nltk.sent_tokenize(synopsis)
        
        # Key theme extraction
        themes = []
        theme_keywords = {
            'love': ['love', 'relationship', 'romance', 'heart'],
            'conflict': ['conflict', 'struggle', 'battle', 'war'],
            'journey': ['journey', 'travel', 'quest', 'adventure'],
            'identity': ['identity', 'self', 'discovery', 'truth'],
            'justice': ['justice', 'right', 'wrong', 'moral']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in synopsis.lower() for keyword in keywords):
                themes.append(theme)
        
        # Emotional tone analysis
        sentiment = self.sentiment_analyzer.polarity_scores(synopsis)
        if sentiment['compound'] > 0.3:
            emotional_tone = 'positive'
        elif sentiment['compound'] < -0.3:
            emotional_tone = 'negative'
        else:
            emotional_tone = 'neutral'
        
        # Complexity assessment
        word_count = len(words)
        if word_count > 200:
            complexity = 'high'
        elif word_count > 100:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        return {
            'length': word_count,
            'sentence_count': len(sentences),
            'avg_sentence_length': round(word_count / max(len(sentences), 1), 1),
            'key_themes': themes[:3],
            'emotional_tone': emotional_tone,
            'complexity': complexity,
            'sentiment_score': round(sentiment['compound'], 2)
        }
    
    def _create_basic_fallback(self, film_data: Dict) -> Dict:
        """Create basic analysis when data is insufficient"""
        return {
            'smart_summary': f"**{film_data.get('title', 'Unknown Film')}** provides a foundation for cinematic exploration with emerging narrative voice.",
            'cinematic_scores': {
                'story_narrative': 2.8,
                'visual_vision': 2.5,
                'technical_craft': 2.3,
                'sound_design': 2.4,
                'performance': 2.7
            },
            'overall_score': 2.7,
            'strengths': ["Foundational concept established", "Clear narrative intention"],
            'weaknesses': ["Limited depth in current form", "Needs further development"],
            'recommendations': ["Expand narrative details", "Develop character depth"],
            'festival_recommendations': {
                'level': 'developing',
                'festivals': ["Local film festivals", "Emerging filmmaker showcases"]
            },
            'audience_analysis': {
                'target_audiences': ["Emerging film enthusiasts", "Workshop audiences"],
                'engagement_score': 0.4,
                'market_potential': 'Developing'
            },
            'genre_insights': {'primary_genre': 'Drama', 'confidence': 60},
            'cultural_insights': {'relevance_score': 0.2, 'is_culturally_relevant': False},
            'scoring_breakdown': {
                'overall_score': 2.7,
                'component_scores': {'narrative': 2.8, 'emotional': 2.6, 'character': 2.7, 'cultural': 2.0, 'technical': 2.4}
            },
            'film_title': film_data.get('title', 'Unknown Film'),
            'philosophical_insights': ["Explores basic human experiences"],
            'ai_tool_suggestions': [],
            'synopsis_analysis': {'length': len(film_data.get('synopsis', '').split()), 'emotional_tone': 'neutral'}
        }
    
    def _create_error_fallback(self, film_data: Dict, error_msg: str) -> Dict:
        """Create error fallback analysis"""
        return {
            'smart_summary': f"**{film_data.get('title', 'Unknown Film')}** encountered analysis challenges. Basic assessment suggests emerging creative potential.",
            'cinematic_scores': {
                'story_narrative': 3.0,
                'visual_vision': 2.8,
                'technical_craft': 2.5,
                'sound_design': 2.6,
                'performance': 2.9
            },
            'overall_score': 2.8,
            'strengths': ["Creative concept identified", "Analysis attempted"],
            'weaknesses': ["Insufficient data for full analysis", f"Technical issue: {error_msg[:50]}"],
            'recommendations': ["Provide more detailed content", "Try manual analysis method"],
            'festival_recommendations': {
                'level': 'developing',
                'festivals': ["Local showcases", "Development workshops"]
            },
            'audience_analysis': {
                'target_audiences': ["Patient early audiences", "Development-focused viewers"],
                'engagement_score': 0.3,
                'market_potential': 'Emerging'
            },
            'genre_insights': {'primary_genre': 'Unknown', 'confidence': 0},
            'cultural_insights': {'relevance_score': 0.0, 'is_culturally_relevant': False},
            'scoring_breakdown': {
                'overall_score': 2.8,
                'component_scores': {'narrative': 3.0, 'emotional': 2.8, 'character': 2.9, 'cultural': 2.2, 'technical': 2.5}
            },
            'film_title': film_data.get('title', 'Unknown Film'),
            'philosophical_insights': ["Analysis in progress"],
            'ai_tool_suggestions': [{'tool': 'Error Recovery', 'purpose': 'Issue diagnosis', 'benefit': 'Improved analysis stability'}],
            'synopsis_analysis': {'length': 0, 'emotional_tone': 'neutral', 'error': error_msg}
        }

# --------------------------
# ENHANCED FILM ANALYSIS INTERFACE WITH VIDEO VIEWER
# --------------------------
class EnhancedFilmAnalysisInterface:
    """Main interface for the film analysis application with integrated video viewer"""
    
    def __init__(self, analyzer: FilmAnalysisEngine):
        self.analyzer = analyzer
        self.persistence = PersistenceManager()
    
    def show_dashboard(self) -> None:
        """Main dashboard for film analysis with enhanced features"""
        st.header("🎬 Fresh-Tomatoes-AI-Film-Meter - Advanced Film Analysis Hub")
        
        # Show enhanced top films section
        self._show_enhanced_top_films_section()
        
        # Check if we should show batch results
        if st.session_state.get('show_batch_results') and st.session_state.get('batch_results'):
            self._display_enhanced_batch_results(st.session_state.batch_results)
            
            if st.button("← Back to Dashboard", key="back_to_dashboard_batch"):
                st.session_state.show_batch_results = False
                st.session_state.batch_results = None
                st.rerun()
            return
        
        # Check if we should show single film results
        if st.session_state.get('show_results_page') and st.session_state.get('current_results_display'):
            self._display_enhanced_film_results(st.session_state.current_results_display)
            
            if st.button("← Back to Dashboard", key="back_to_dashboard"):
                st.session_state.show_results_page = False
                st.session_state.current_results_display = None
                st.rerun()
            return
        
        # Display enhanced statistics
        stats = self._get_enhanced_statistics()
        
        st.subheader("📊 Advanced Analytics Dashboard")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Total Films", stats['total_films'], help="Total number of films analyzed")
        with col2:
            if stats['total_films'] > 0:
                st.metric("Average Score", f"{stats['average_score']}/5.0",
                         delta=f"{stats['score_trend']:.1f} trend", 
                         delta_color="normal" if stats['score_trend'] >= 0 else "inverse",
                         help="Average score with trend analysis")
        with col3:
            if stats['total_films'] > 0:
                st.metric("Score Range", f"{stats['score_range']}", help="Difference between highest and lowest scores")
        with col4:
            if stats['total_films'] > 0:
                st.metric("Cultural Films", stats['cultural_films'], help="Films with significant cultural relevance")
        with col5:
            if stats['total_films'] > 0:
                st.metric("Top Genre", stats['top_genre'][:15], help="Most frequently detected genre")
        with col6:
            if stats['total_films'] > 0:
                st.metric("Analysis Rate", f"{stats['analysis_rate']}/day", help="Average analyses per day")
        
        # Quick insights panel
        with st.expander("💡 **Quick Insights & Trends**", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📈 Recent Activity:**")
                if stats['recent_analyses']:
                    for analysis in stats['recent_analyses'][:3]:
                        st.write(f"• {analysis['title'][:20]}: {analysis['score']}/5.0")
                else:
                    st.write("No recent analyses")
            
            with col2:
                st.write("**🎭 Genre Distribution:**")
                if stats['genre_distribution']:
                    for genre, count in list(stats['genre_distribution'].items())[:3]:
                        st.write(f"• {genre}: {count}")
                else:
                    st.write("No genre data")
        
        # Analysis methods tabs
        st.subheader("🎬 Analyze Films")
        tab1, tab2, tab3 = st.tabs(["🎥 YouTube Analysis", "📝 Manual Entry", "📊 CSV Batch"])
        
        with tab1:
            self._show_youtube_analysis()
        with tab2:
            self._show_manual_analysis()
        with tab3:
            self._show_csv_interface()
    
    def _show_youtube_analysis(self) -> None:
        """Show YouTube video analysis interface"""
        st.subheader("🎥 YouTube Video Analysis")
        st.markdown("Analyze films from YouTube videos by providing a video URL or ID.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            youtube_input = st.text_input(
                "Enter YouTube URL or Video ID:",
                placeholder="https://www.youtube.com/watch?v= ... or just the video ID",
                key="youtube_input"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            analyze_button = st.button("🎬 Analyze Video", type="primary", width='stretch')
        
        if analyze_button and youtube_input:
            with st.spinner("🔄 Extracting and analyzing video content..."):
                try:
                    # Extract video ID from URL
                    video_id = self._extract_youtube_id(youtube_input)
                    
                    if not video_id:
                        st.error("Invalid YouTube URL or Video ID. Please check your input.")
                        return
                    
                    # Try to get video info first
                    video_info = self._get_youtube_video_info(video_id)
                    
                    # Get video transcript
                    transcript_text = ""
                    try:
                        # Use correct YouTubeTranscriptApi method
                        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                        
                        # Try to get transcript in preferred language order
                        languages = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
                        transcript_found = False
                        
                        for lang in languages:
                            try:
                                transcript = transcript_list.find_transcript([lang])
                                transcript_data = transcript.fetch()
                                transcript_text = " ".join([entry['text'] for entry in transcript_data])
                                transcript_found = True
                                st.info(f"✅ Found transcript in {lang}")
                                break
                            except:
                                continue
                        
                        if not transcript_found:
                            # Try any available transcript
                            try:
                                # Get first available transcript
                                for transcript in transcript_list:
                                    transcript_data = transcript.fetch()
                                    transcript_text = " ".join([entry['text'] for entry in transcript_data])
                                    st.info(f"✅ Found transcript in {transcript.language}")
                                    break
                            except Exception as e:
                                st.warning(f"⚠️ Could not extract transcript: {str(e)[:100]}")
                                transcript_text = ""
                                
                    except Exception as e:
                        st.warning(f"⚠️ Could not extract transcript: {str(e)[:100]}")
                        st.info("Continuing analysis with available metadata only...")
                    
                    # Prepare film data
                    film_data = {
                        'title': video_info.get('title', 'YouTube Video'),
                        'synopsis': f"YouTube video: {video_info.get('description', '')[:500]}...",
                        'transcript': transcript_text,
                        'video_id': video_id,
                        'video_title': video_info.get('title', 'Unknown'),
                        'duration': self._format_duration(video_info.get('duration', 0)),
                        'channel': video_info.get('channel', 'Unknown'),
                        'views': video_info.get('views', 0),
                        'upload_date': video_info.get('upload_date', ''),
                        'source': 'youtube'
                    }
                    
                    # If no transcript, provide guidance
                    if not transcript_text.strip():
                        st.warning("No transcript available. Analysis will be based on video metadata only.")
                        film_data['synopsis'] = f"YouTube video by {film_data['channel']}: {video_info.get('description', 'No description available')[:300]}"
                    
                    # Analyze the film
                    results = self.analyzer.analyze_film(film_data)
                    
                    # Store video info in session state
                    st.session_state.current_video_id = video_id
                    st.session_state.current_video_title = video_info.get('title', 'Unknown')
                    
                    # Display success message
                    st.success(f"✅ Successfully analyzed: {video_info.get('title', 'Unknown')}")
                    
                    # Set results to display
                    st.session_state.current_results_display = results
                    st.session_state.show_results_page = True
                    
                    # Rerun to show results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error analyzing video: {str(e)}")
        
        # Show help and examples
        with st.expander("ℹ️ How to use YouTube Analysis", expanded=False):
            st.markdown("""
            **Instructions:**
            1. Paste a YouTube URL or Video ID
            2. Click "Analyze Video"
            3. Wait for transcript extraction
            4. View comprehensive analysis with embedded video viewer
            
            **Examples:**
            - Full URL: `https://www.youtube.com/watch?v=dQw4w9WgXcQ `
            - Short URL: `https://youtu.be/dQw4w9WgXcQ `
            - Just the ID: `dQw4w9WgXcQ`
            
            **Note:** Not all YouTube videos have transcripts available. 
            If transcript extraction fails, try the manual entry method.
            """)
    
    def _show_manual_analysis(self) -> None:
        """Show manual film analysis interface"""
        st.subheader("📝 Manual Film Analysis")
        st.markdown("Enter film details manually for comprehensive analysis.")
        
        with st.form("manual_analysis_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Film Title *", placeholder="Enter film title", key="manual_title")
                director = st.text_input("Director", placeholder="Director's name", key="manual_director")
                writer = st.text_input("Writer", placeholder="Writer's name", key="manual_writer")
            
            with col2:
                duration = st.text_input("Duration", placeholder="e.g., 120m, 2h 15m", key="manual_duration")
                genre = st.text_input("Genre (optional)", placeholder="e.g., Drama, Comedy", key="manual_genre")
                year = st.number_input("Year", min_value=1900, max_value=2100, value=2023, key="manual_year")
            
            synopsis = st.text_area(
                "Synopsis/Description *", 
                placeholder="Enter a detailed synopsis of the film...",
                height=150,
                key="manual_synopsis"
            )
            
            transcript = st.text_area(
                "Transcript/Dialogue (optional)", 
                placeholder="Paste film transcript, dialogue, or key scenes...",
                height=200,
                key="manual_transcript"
            )
            
            submitted = st.form_submit_button("🎬 Analyze Film", type="primary", width='stretch')
            
            if submitted:
                if not title or not synopsis:
                    st.error("Please provide at least a film title and synopsis.")
                    return
                
                # Prepare film data
                film_data = {
                    'title': title,
                    'director': director or "Unknown",
                    'writer': writer or "Unknown",
                    'duration': duration or "Unknown",
                    'genre': genre or "",
                    'year': year,
                    'synopsis': synopsis,
                    'transcript': transcript,
                    'source': 'manual'
                }
                
                # Analyze the film
                with st.spinner("🔍 Analyzing film content..."):
                    results = self.analyzer.analyze_film(film_data)
                    
                    # Display success message
                    st.success(f"✅ Successfully analyzed: {title}")
                    
                    # Set results to display
                    st.session_state.current_results_display = results
                    st.session_state.show_results_page = True
                    
                    # Rerun to show results
                    st.rerun()
    
    def _show_csv_interface(self) -> None:
        """Show CSV batch analysis interface"""
        st.subheader("📊 Batch Analysis via CSV")
        st.markdown("Upload a CSV file to analyze multiple films at once.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="CSV should contain columns: title, synopsis (optional: director, writer, duration, genre, year, transcript)"
            )
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            analyze_batch = st.button("📦 Analyze Batch", type="primary", width='stretch', disabled=not uploaded_file)
        
        if uploaded_file and analyze_batch:
            with st.spinner("📊 Processing batch analysis..."):
                try:
                    # Read CSV file
                    df = pd.read_csv(uploaded_file)
                    
                    # Check required columns
                    if 'title' not in df.columns or 'synopsis' not in df.columns:
                        st.error("CSV must contain 'title' and 'synopsis' columns.")
                        return
                    
                    # Process each row
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        # Prepare film data
                        film_data = {
                            'title': str(row.get('title', 'Unknown Film')),
                            'synopsis': str(row.get('synopsis', '')),
                            'director': str(row.get('director', 'Unknown')),
                            'writer': str(row.get('writer', 'Unknown')),
                            'duration': str(row.get('duration', 'Unknown')),
                            'genre': str(row.get('genre', '')),
                            'year': int(row.get('year', 2023)) if pd.notna(row.get('year')) else 2023,
                            'transcript': str(row.get('transcript', '')),
                            'source': 'csv_batch'
                        }
                        
                        # Analyze film
                        analysis_result = self.analyzer.analyze_film(film_data)
                        
                        # Store results
                        results.append({
                            'title': film_data['title'],
                            'overall_score': analysis_result['overall_score'],
                            'genre': analysis_result.get('genre_insights', {}).get('primary_genre', 'Unknown'),
                            'cultural_relevance': analysis_result.get('cultural_insights', {}).get('relevance_score', 0),
                            'analysis_result': analysis_result,
                            'film_data': film_data
                        })
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(df))
                    
                    # Store batch results
                    st.session_state.batch_results = results
                    st.session_state.show_batch_results = True
                    
                    # Show success message
                    st.success(f"✅ Successfully analyzed {len(results)} films!")
                    st.info(f"📈 Average score: {np.mean([r['overall_score'] for r in results]):.1f}/5.0")
                    
                    # Rerun to show results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing CSV: {str(e)}")
        
        # Show CSV template and instructions
        with st.expander("📋 CSV Format Instructions", expanded=False):
            st.markdown("""
            **Required columns:**
            - `title`: Film title (string)
            - `synopsis`: Film description/summary (string)
            
            **Optional columns:**
            - `director`: Director's name (string)
            - `writer`: Writer's name (string) 
            - `duration`: Film duration (string, e.g., "120m", "2h 15m")
            - `genre`: Film genre (string)
            - `year`: Release year (integer)
            - `transcript`: Full transcript or key dialogue (string)
            
            **Example CSV format:**
            ```csv
            title,synopsis,director,genre,year
            "Urban Dreams","A story about city life...","John Doe","Drama",2023
            "Concrete Memories","Exploring urban identity...","Jane Smith","Documentary",2022
            ```
            
            **Note:** 
            - CSV should have a header row
            - Maximum recommended batch size: 50 films
            - Analysis time depends on content length
            """)
    
    def _extract_youtube_id(self, url_or_id: str) -> Optional[str]:
        """Extract YouTube video ID from URL or return as-is if already an ID"""
        # If it looks like just an ID (no special characters except dash and underscore)
        if re.match(r'^[\w\-_]{11}$', url_or_id):
            return url_or_id
        
        # Try to extract from various YouTube URL formats
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([\w\-_]{11})',
            r'(?:youtu\.be\/)([\w\-_]{11})',
            r'(?:youtube\.com\/embed\/)([\w\-_]{11})',
            r'(?:youtube\.com\/v\/)([\w\-_]{11})',
            r'(?:youtube\.com\/watch\?.*v=)([\w\-_]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        return None
    
    def _get_youtube_video_info(self, video_id: str) -> Dict:
        """Get YouTube video information (simulated or using oEmbed API)"""
        try:
            # Try to get info from YouTube oEmbed API
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(oembed_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', 'YouTube Video'),
                    'description': data.get('author_name', '') + ' - YouTube video',
                    'duration': 0,  # oEmbed doesn't provide duration
                    'channel': data.get('author_name', 'Unknown Channel'),
                    'views': 0,
                    'upload_date': '',
                    'thumbnail_url': data.get('thumbnail_url', f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg')
                }
        
        except Exception as e:
            print(f"Could not fetch video info from API: {str(e)}")
        
        # Fallback to simulated data
        film_titles = [
            "Urban Dreams: A City Story",
            "The Last Sunset",
            "Echoes of Tomorrow",
            "Shadows in the City",
            "Voices from the Street",
            "Concrete Dreams",
            "The Neighborhood Chronicles",
            "City Lights, Dark Nights"
        ]
        
        descriptions = [
            "A compelling story about urban life and personal struggles.",
            "Exploring themes of identity and community in modern society.",
            "A film that captures the essence of contemporary challenges.",
            "Storytelling that reflects on human connections in a digital age."
        ]
        
        channels = [
            "Independent Filmmaker",
            "Urban Cinema Collective",
            "Digital Storytellers",
            "Film Festival Selection"
        ]
        
        return {
            'title': random.choice(film_titles),
            'description': random.choice(descriptions),
            'duration': random.randint(120, 1800),
            'channel': random.choice(channels),
            'views': random.randint(1000, 1000000),
            'upload_date': f"202{random.randint(2, 4)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            'thumbnail_url': f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        }
    
    def _format_duration(self, seconds: int) -> str:
        """Format duration in seconds to readable format"""
        if not seconds:
            return "Unknown"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {secs}s"
    
    def _show_enhanced_top_films_section(self) -> None:
        """Show enhanced top films section with philosophical insights"""
        top_films = self.persistence.get_top_films()
        
        if top_films:
            st.subheader("🏆 Top Films - Cinematic Excellence")
            st.caption("*Films demonstrating exceptional artistic merit and cultural resonance*")
            
            cols = st.columns(min(3, len(top_films)))
            
            for idx, film_data in enumerate(top_films[:3]):
                with cols[idx]:
                    analysis = film_data['analysis_results']
                    film_info = film_data['film_data']
                    
                    # Extract enhanced info
                    genre = analysis.get('genre_insights', {}).get('primary_genre', 'Unknown')
                    if isinstance(genre, dict):
                        genre = genre.get('primary_genre', 'Unknown')
                    
                    cultural_score = analysis.get('cultural_insights', {}).get('relevance_score', 0)
                    philosophical = analysis.get('philosophical_insights', [])
                    philosophical_text = philosophical[0] if philosophical else "Artistic expression"
                    
                    # Enhanced film card
                    cultural_badge = "🌍" if cultural_score > 0.5 else ""
                    philosophical_icon = "💭" if philosophical else "🎨"
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              border-radius: 10px; padding: 15px; margin: 10px 0; border: 2px solid gold;
                              box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <h4 style='color: white; margin: 0 0 10px 0; text-align: center;'>{film_info.get('title', 'Unknown')[:25]}</h4>
                        <div style='text-align: center;'>
                            <h1 style='color: gold; margin: 5px 0; font-size: 32px;'>{analysis['overall_score']}/5.0</h1>
                            <p style='color: white; margin: 5px 0; font-size: 14px;'>
                                {genre} {cultural_badge} {philosophical_icon}
                            </p>
                            <p style='color: #ddd; margin: 5px 0; font-size: 12px;'>
                                {film_info.get('director', 'Unknown')[:20]}
                            </p>
                            <p style='color: #ccc; margin: 5px 0; font-size: 11px; font-style: italic;'>
                                "{philosophical_text[:60]}..."
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced view button with philosophical context
                    if st.button(f"🔍 Deep Analysis", key=f"view_top_{idx}", width='stretch'):
                        st.session_state.current_results_display = analysis
                        st.session_state.current_analysis_id = film_data.get('film_id')
                        st.session_state.show_results_page = True
                        st.rerun()
        
        st.markdown("---")
    
    def _display_enhanced_batch_results(self, batch_results: List[Dict]) -> None:
        """Display enhanced batch analysis results"""
        st.header("📊 Batch Analysis Results")
        
        if not batch_results:
            st.info("No batch results to display.")
            return
        
        # Summary statistics
        total_films = len(batch_results)
        scores = [r['overall_score'] for r in batch_results]
        avg_score = np.mean(scores)
        cultural_scores = [r['cultural_relevance'] for r in batch_results]
        avg_cultural = np.mean(cultural_scores)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Films", total_films)
        with col2:
            st.metric("Average Score", f"{avg_score:.1f}/5.0")
        with col3:
            st.metric("High Scores (≥4.0)", sum(1 for s in scores if s >= 4.0))
        with col4:
            st.metric("Avg Cultural", f"{avg_cultural:.0%}")
        
        # Results table
        st.subheader("📋 Detailed Results")
        
        # Create results dataframe
        results_data = []
        for result in batch_results:
            results_data.append({
                'Title': result['title'][:30],
                'Score': result['overall_score'],
                'Genre': result['genre'][:15] if isinstance(result['genre'], str) else 'Unknown',
                'Cultural': f"{result['cultural_relevance']:.0%}",
                'Status': '🏆 Top' if result['overall_score'] >= 4.0 else '✅ Good' if result['overall_score'] >= 3.0 else '📈 Developing'
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Score distribution
        st.subheader("📈 Score Distribution")
        
        fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=10, marker_color='#667eea')])
        fig.update_layout(
            title='Score Distribution',
            xaxis_title='Score',
            yaxis_title='Count',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top films section
        st.subheader("🏆 Top Performing Films")
        
        top_films = sorted(batch_results, key=lambda x: x['overall_score'], reverse=True)[:3]
        
        for i, film in enumerate(top_films):
            with st.expander(f"{i+1}. {film['title'][:30]} - {film['overall_score']}/5.0"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Score:** {film['overall_score']}/5.0")
                    st.write(f"**Genre:** {film['genre']}")
                    st.write(f"**Cultural Relevance:** {film['cultural_relevance']:.0%}")
                
                with col2:
                    if st.button("View Full Analysis", key=f"batch_view_{i}"):
                        st.session_state.current_results_display = film['analysis_result']
                        st.session_state.show_results_page = True
                        st.session_state.batch_results = None
                        st.session_state.show_batch_results = False
                        st.rerun()
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download as CSV", width='stretch'):
                # Create downloadable CSV
                export_df = pd.DataFrame([
                    {
                        'title': r['title'],
                        'score': r['overall_score'],
                        'genre': r['genre'],
                        'cultural_relevance': r['cultural_relevance'],
                        'director': r['film_data'].get('director', ''),
                        'year': r['film_data'].get('year', ''),
                        'analysis_summary': r['analysis_result'].get('smart_summary', '')[:200]
                    }
                    for r in batch_results
                ])
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Click to download",
                    data=csv,
                    file_name=f"film_analysis_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch'
                )
        
        with col2:
            if st.button("✨ Save to Projects", width='stretch'):
                project_name = st.text_input("Project Name:", value=f"Batch Analysis {datetime.now().strftime('%Y-%m-%d')}")
                
                if project_name:
                    for result in batch_results:
                        self.persistence.save_project(
                            f"{project_name} - {result['title'][:20]}",
                            result['film_data'],
                            result['analysis_result']
                        )
                    st.success(f"✅ Saved {len(batch_results)} films to project: {project_name}")
    
    def _get_enhanced_statistics(self) -> Dict:
        """Get enhanced analysis statistics"""
        films = list(st.session_state.stored_results.values())
        history = st.session_state.analysis_history
        
        if not films:
            return {
                "total_films": 0,
                "average_score": 0,
                "score_range": 0,
                "cultural_films": 0,
                "top_genre": "N/A",
                "analysis_rate": 0,
                "score_trend": 0,
                "recent_analyses": [],
                "genre_distribution": {}
            }
        
        scores = [film["analysis_results"]["overall_score"] for film in films]
        cultural_films = sum(1 for film in films 
                           if film["analysis_results"].get('cultural_insights', {}).get('is_culturally_relevant', False))
        
        # Genre distribution
        genre_counter = Counter()
        for item in history:
            genre = item.get('detected_genre', 'Unknown')
            if genre and genre != 'Unknown':
                genre_counter[genre] += 1
        
        # Score trend (last 5 vs previous 5)
        trend = 0
        if len(history) >= 10:
            recent_scores = [h['overall_score'] for h in history[-5:]]
            previous_scores = [h['overall_score'] for h in history[-10:-5]]
            if previous_scores:
                trend = np.mean(recent_scores) - np.mean(previous_scores)
        
        # Analysis rate (analyses per day)
        if len(history) > 1:
            dates = [datetime.fromisoformat(h['timestamp']).date() for h in history]
            date_range = (max(dates) - min(dates)).days or 1
            analysis_rate = len(history) / max(date_range, 1)
        else:
            analysis_rate = 0
        
        # Recent analyses
        recent_analyses = []
        for item in history[-5:]:
            recent_analyses.append({
                'title': item.get('title', 'Unknown'),
                'score': item.get('overall_score', 0),
                'genre': item.get('detected_genre', 'Unknown')
            })
        
        return {
            "total_films": len(films),
            "average_score": round(np.mean(scores), 2),
            "highest_score": round(max(scores), 2),
            "lowest_score": round(min(scores), 2),
            "score_range": round(max(scores) - min(scores), 2),
            "score_std": round(np.std(scores), 2) if len(scores) > 1 else 0,
            "cultural_films": cultural_films,
            "top_genre": genre_counter.most_common(1)[0][0] if genre_counter else "N/A",
            "analysis_rate": round(analysis_rate, 1),
            "score_trend": round(trend, 2),
            "recent_analyses": recent_analyses[::-1],
            "genre_distribution": dict(genre_counter.most_common(5))
        }

    # -------------- COMPLETE MISSING METHOD --------------
    def _display_enhanced_film_results(self, results: Dict) -> None:
        """Display enhanced film analysis results with integrated video viewer"""
        st.success("🎉 Advanced Film Analysis Complete!")

        # Get film data
        film_data = {}
        if st.session_state.current_analysis_id:
            stored_result = self.persistence.load_results(st.session_state.current_analysis_id)
            if stored_result:
                film_data = stored_result['film_data']

        # Display film title with philosophical context
        film_title = film_data.get('title', results.get('film_title', 'Unknown Film'))
        philosophical_insights = results.get('philosophical_insights', [])
        primary_insight = philosophical_insights[0] if philosophical_insights else "Cinematic Exploration"

        # ============================================
        # VIDEO VIEWER SECTION - EMBEDDED YOUTUBE PLAYER
        # ============================================

        # Check if we have a YouTube video to display
        video_id = None
        video_title = None

        # Check session state first
        if st.session_state.get('current_video_id'):
            video_id = st.session_state.current_video_id
            video_title = st.session_state.get('current_video_title', '')
        # Check film data
        elif film_data.get('video_id'):
            video_id = film_data['video_id']
            video_title = film_data.get('video_title', '')

        # If we have a video ID, create a video viewer section
        if video_id:
            st.markdown("---")
            st.subheader("🎬 Film / Video Viewer")

            # Create a two-column layout for video and info
            video_col, info_col = st.columns([3, 2])

            with video_col:
                # Display YouTube embed
                embed_html = f"""
                <div style="border-radius: 10px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.2); margin-bottom: 15px;">
                    <iframe 
                        width="100%" 
                        height="400" 
                        src="https://www.youtube.com/embed/{video_id}?rel=0&modestbranding=1" 
                        title="YouTube video player" 
                        frameborder="0" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen
                        style="border-radius: 10px;">
                    </iframe>
                </div>
                """
                st.markdown(embed_html, unsafe_allow_html=True)

                # Video controls
                st.caption("🎥 Use the video player to watch scenes and verify analysis")

            with info_col:
                st.markdown("**📺 Video Details**")
                if video_title:
                    st.write(f"**Title:** {video_title}")

                # Show video duration if available
                if film_data.get('duration'):
                    st.write(f"**Duration:** {film_data.get('duration')}")

                # Show channel info if available
                if film_data.get('channel'):
                    st.write(f"**Channel:** {film_data.get('channel')}")

                if film_data.get('views'):
                    st.write(f"**Views:** {film_data.get('views'):,}")

                if film_data.get('upload_date'):
                    st.write(f"**Uploaded:** {film_data.get('upload_date')}")

                # Quick actions
                st.markdown("---")
                st.markdown("**🔗 Quick Links**")

                # Create buttons for YouTube actions
                yt_col1, yt_col2 = st.columns(2)
                with yt_col1:
                    youtube_url = f"https://youtube.com/watch?v={video_id}"
                    if st.button("📺 Open YouTube", key="open_yt", width='stretch'):
                        st.markdown(f'<meta http-equiv="refresh" content="0; url={youtube_url}">', unsafe_allow_html=True)

                with yt_col2:
                    if st.button("📋 Copy Link", key="copy_yt", width='stretch'):
                        st.code(youtube_url, language="text")
                        st.toast("YouTube link copied!", icon="✅")

            st.markdown("---")

        # ============================================
        # END OF VIDEO VIEWER SECTION
        # ============================================

        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  border-radius: 15px; margin: 20px 0; border: 3px solid white; box-shadow: 0 6px 12px rgba(0,0,0,0.15);'>
            <h1 style='color: white; margin: 0; font-size: 32px;'>{film_title}</h1>
            <p style='color: #ddd; margin: 10px 0 0 0; font-size: 16px; font-style: italic;'>
                "{primary_insight}"
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Overall score with contextual explanation
        overall_score = results['overall_score']
        score_context = self._get_score_context(overall_score)

        st.markdown(f"""
        <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  border-radius: 15px; margin: 20px 0; border: 3px solid #FFD700; box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
            <h1 style='color: gold; margin: 0; font-size: 48px;'>{overall_score}/5.0</h1>
            <p style='color: white; font-size: 20px; margin: 10px 0;'>🎬 Cinematic Score</p>
            <p style='color: #eee; font-size: 14px; margin: 5px 0;'>{score_context}</p>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced Film Details Section
        st.subheader("📋 Comprehensive Film Analysis")

        # Create tabs for different analysis aspects
        # Add "🎥 Video Analysis" tab if we have video
        if video_id:
            detail_tabs = st.tabs(["🎬 Film Info", "🎥 Video Analysis", "🧠 Philosophical", "🤖 AI Tools", "📊 Analytics"])
        else:
            detail_tabs = st.tabs(["🎬 Film Information", "🧠 Philosophical Insights", "🤖 AI Enhancement", "📊 Deep Analytics"])

        if video_id:
            # With video tabs
            with detail_tabs[0]:
                self._display_film_information(film_data, results)

            with detail_tabs[1]:
                self._display_video_analysis_section(film_data, results, video_id)

            with detail_tabs[2]:
                self._display_philosophical_insights(results)

            with detail_tabs[3]:
                self._display_ai_enhancements(results)

            with detail_tabs[4]:
                self._display_deep_analytics(results)
        else:
            # Original tabs (no video)
            with detail_tabs[0]:
                self._display_film_information(film_data, results)

            with detail_tabs[1]:
                self._display_philosophical_insights(results)

            with detail_tabs[2]:
                self._display_ai_enhancements(results)

            with detail_tabs[3]:
                self._display_deep_analytics(results)

        # Enhanced Category Scores with Visualizations
        st.subheader("🎯 Multidimensional Analysis")

        scores = results['cinematic_scores']

        # Create a radar chart for scores
        self._create_score_radar_chart(scores)

        # Score breakdown expander
        with st.expander("📈 **Advanced Score Breakdown & Distribution**", expanded=st.session_state.get('show_breakdown', False)):
            self._display_score_breakdown(results)

        # Synopsis Analysis
        synopsis_analysis = results.get('synopsis_analysis', {})
        if synopsis_analysis.get('length', 0) > 0:
            st.subheader("📖 Synopsis Analysis")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Length", f"{synopsis_analysis['length']} words")
            with col2:
                st.metric("Sentiment", synopsis_analysis['emotional_tone'].title())
            with col3:
                st.metric("Complexity", synopsis_analysis['complexity'].title())

            if synopsis_analysis.get('key_themes'):
                st.write("**Key Themes:** " + ", ".join(synopsis_analysis['key_themes']))

        # Recommendations and Next Steps
        st.subheader("🚀 Strategic Recommendations")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**🎪 Festival Strategy:**")
            festival_recs = results['festival_recommendations']
            st.write(f"**Level:** {festival_recs['level']}")
            for festival in festival_recs['festivals']:
                st.write(f"• {festival}")

        with col2:
            st.write("**🎯 Development Path:**")
            for recommendation in results.get('recommendations', []):
                st.write(f"• {recommendation}")

    def _get_score_context(self, score: float) -> str:
        """Get contextual explanation for score"""
        if score >= 4.5:
            return "Exceptional - Award-caliber cinematic achievement"
        elif score >= 4.0:
            return "Excellent - Professional quality with strong artistic vision"
        elif score >= 3.5:
            return "Strong - Compelling work with clear potential"
        elif score >= 3.0:
            return "Solid - Well-executed foundation for development"
        elif score >= 2.5:
            return "Developing - Promising concepts with room for growth"
        else:
            return "Emerging - Foundational creative exploration"

    def _display_film_information(self, film_data: Dict, results: Dict) -> None:
        """Display comprehensive film information"""
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎬 Film Information**")
            st.write(f"**Title:** {film_data.get('title', 'Unknown')}")
            if film_data.get('director') and film_data.get('director') != 'Unknown':
                st.write(f"**Director:** {film_data.get('director')}")
            if film_data.get('writer') and film_data.get('writer') != 'Unknown':
                st.write(f"**Writer:** {film_data.get('writer')}")
            if film_data.get('duration') and film_data.get('duration') != 'Unknown':
                st.write(f"**Duration:** {film_data.get('duration')}")

            # Synopsis preview
            if film_data.get('synopsis'):
                with st.expander("📖 View Synopsis", expanded=False):
                    st.write(film_data.get('synopsis'))

        with col2:
            st.markdown("**📊 Analysis Details**")

            genre_insights = results['genre_insights']
            if isinstance(genre_insights, dict) and 'primary_genre' in genre_insights:
                st.write(f"**Primary Genre:** {genre_insights['primary_genre']}")
                if genre_insights.get('confidence'):
                    st.write(f"**Confidence:** {genre_insights['confidence']}%")
                if genre_insights.get('secondary_genres'):
                    st.write(f"**Secondary Genres:** {', '.join(genre_insights['secondary_genres'])}")
            else:
                st.write(f"**Detected Genre:** {genre_insights.get('detected_genre', 'Unknown')}")

            cultural_insights = results.get('cultural_insights', {})
            if cultural_insights.get('is_culturally_relevant'):
                relevance = cultural_insights.get('relevance_score', 0)
                st.write(f"**Cultural Relevance:** {relevance:.0%}")
                if cultural_insights.get('primary_themes'):
                    st.write(f"**Cultural Themes:** {', '.join(cultural_insights['primary_themes'])}")

            if st.session_state.last_analysis_time:
                last_time = datetime.fromisoformat(st.session_state.last_analysis_time)
                st.write(f"**Analyzed:** {last_time.strftime('%Y-%m-%d %H:%M')}")

    def _display_video_analysis_section(self, film_data: Dict, results: Dict, video_id: str) -> None:
        """Display video-specific analysis section"""
        st.markdown("### 🎥 Video Content Analysis")

        # Transcript analysis
        transcript = film_data.get('transcript', '')
        if transcript:
            word_count = len(transcript.split())
            st.write(f"**Transcript Analysis:** {word_count} words")

            # Key moments
            with st.expander("🔍 View Key Video Moments", expanded=False):
                if word_count > 500:
                    # Extract sample lines from transcript
                    lines = transcript.split('.')
                    key_lines = [line.strip() for line in lines[:10] if len(line.strip()) > 50]
                    for i, line in enumerate(key_lines[:5]):
                        st.write(f"**Moment {i+1}:** {line[:200]}...")
                else:
                    st.write("Transcript preview:")
                    st.text(transcript[:500] + "..." if len(transcript) > 500 else transcript)

            # Sentiment from transcript
            if transcript:
                sentiment = self.analyzer.sentiment_analyzer.polarity_scores(transcript)
                st.write(f"**Transcript Sentiment:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive", f"{sentiment['pos']:.0%}")
                with col2:
                    st.metric("Neutral", f"{sentiment['neu']:.0%}")
                with col3:
                    st.metric("Negative", f"{sentiment['neg']:.0%}")
        else:
            st.info("No transcript available for this video. Score based on metadata only.")

        # Study Group Helper
        st.markdown("---")
        st.markdown("### 👥 Study Group Helper")

        st.write("**Watch these key moments to understand the scoring:**")

        # Create key moments based on scoring
        score_moments = []
        cinematic_scores = results['cinematic_scores']

        if cinematic_scores.get('performance', 0) >= 4.0:
            score_moments.append("Watch character interactions and performances (around 25% mark)")

        if cinematic_scores.get('story_narrative', 0) >= 4.0:
            score_moments.append("Observe narrative structure and plot development (mid-point)")

        if cinematic_scores.get('visual_vision', 0) >= 4.0:
            score_moments.append("Notice cinematography and visual composition (throughout)")

        if cinematic_scores.get('sound_design', 0) >= 4.0:
            score_moments.append("Listen to audio design and music integration (climax scene)")

        # Add default moments if none specific
        if not score_moments:
            score_moments = [
                "Watch opening scene for establishing tone",
                "Observe character development in middle act",
                "Notice resolution and ending impact"
            ]

        for i, moment in enumerate(score_moments[:4]):
            st.write(f"**{i+1}.** {moment}")

        # Video analysis tips
        with st.expander("💡 Video Analysis Tips", expanded=False):
            st.markdown("""
            **For Study Groups:**
            1. **Pause and Discuss:** Stop at key moments to discuss scoring criteria
            2. **Scene Comparison:** Compare different scenes against score components
            3. **Group Scoring:** Have each member score independently, then compare
            4. **Cultural Context:** Discuss how cultural elements affect scoring

            **Video Controls:**
            - Use YouTube's speed controls for detailed analysis
            - Turn on captions for dialogue analysis
            - Take timestamp notes for specific moments
            """)

    def _display_philosophical_insights(self, results: Dict) -> None:
        """Display philosophical insights"""
        philosophical_insights = results.get('philosophical_insights', [])
        cultural_insights = results.get('cultural_insights', {})

        if philosophical_insights or cultural_insights.get('philosophical_insights'):
            st.write("**💭 Philosophical Framework:**")

            all_insights = []
            if philosophical_insights:
                all_insights.extend(philosophical_insights)
            if cultural_insights.get('philosophical_insights'):
                all_insights.extend(cultural_insights['philosophical_insights'])

            for insight in all_insights[:3]:
                st.write(f"• {insight}")

            # Genre philosophical aspect
            genre_details = results.get('genre_insights', {}).get('details', {})
            if genre_details and 'philosophical_aspect' in genre_details:
                st.write(f"\n**🎭 Genre Philosophy:** {genre_details['philosophical_aspect']}")
        else:
            st.info("No specific philosophical insights detected. This film appears to focus on direct narrative storytelling.")

    def _display_ai_enhancements(self, results: Dict) -> None:
        """Display AI tool suggestions for enhancement"""
        ai_suggestions = results.get('ai_tool_suggestions', [])

        if ai_suggestions:
            st.write("**🤖 AI Enhancement Opportunities:**")
            st.caption("Suggested tools for deeper analysis and improved scoring")

            for suggestion in ai_suggestions:
                with st.expander(f"{suggestion['tool']} - {suggestion['purpose']}"):
                    st.write(f"**Purpose:** {suggestion['purpose']}")
                    st.write(f"**Benefit:** {suggestion['benefit']}")
                    st.write(f"**Implementation:** Could enhance scoring accuracy by 10-15%")
        else:
            st.info("Current analysis provides comprehensive coverage. For advanced needs, consider GPT-4 for narrative analysis or BERT for cultural context.")

    def _display_deep_analytics(self, results: Dict) -> None:
        """Display deep analytics and metrics"""
        scoring_breakdown = results.get('scoring_breakdown', {})
        component_scores = scoring_breakdown.get('component_scores', {})
        weights = scoring_breakdown.get('applied_weights', {})

        if component_scores and weights:
            st.write("**📊 Scoring Algorithm Details:**")

            # Create a DataFrame for visualization
            score_data = []
            for component, score in component_scores.items():
                weight = weights.get(component, 0)
                weighted_score = score * weight
                score_data.append({
                    'Component': component.title(),
                    'Score': score,
                    'Weight': weight,
                    'Weighted': round(weighted_score, 2)
                })

            df = pd.DataFrame(score_data)
            st.dataframe(df, use_container_width=True)

            # Cultural bonus
            cultural_bonus = scoring_breakdown.get('cultural_bonus', 0)
            if cultural_bonus > 0:
                st.success(f"🎉 **Cultural Bonus Applied:** +{cultural_bonus:.3f} points")

    def _create_score_radar_chart(self, scores: Dict) -> None:
        """Create a radar chart for cinematic scores"""
        try:
            categories = list(scores.keys())
            values = list(scores.values())

            fig = go.Figure(data=go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=[cat.replace('_', ' ').title() for cat in categories] + [categories[0].replace('_', ' ').title()],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='rgb(102, 126, 234)', width=2),
                hoverinfo='text',
                text=[f"{cat.replace('_', ' ').title()}: {val}/5.0" for cat, val in zip(categories, values)]
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5],
                        tickvals=[1, 2, 3, 4, 5],
                        ticktext=['1', '2', '3', '4', '5']
                    )
                ),
                showlegend=False,
                height=300,
                margin=dict(l=50, r=50, t=30, b=30)
            )

            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            # Fallback to simple columns if plotly not available
            cols = st.columns(len(scores))
            categories = [
                ("🧠 Story", scores['story_narrative'], "#FF6B6B"),
                ("👁️ Visual", scores['visual_vision'], "#4ECDC4"),
                ("⚡ Technical", scores['technical_craft'], "#45B7D1"),
                ("🎵 Sound", scores['sound_design'], "#96CEB4"),
                ("🌟 Performance", scores['performance'], "#FFD93D")
            ]

            for idx, (name, score, color) in enumerate(categories):
                with cols[idx]:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: {color}; 
                              border-radius: 10px; margin: 5px; border: 2px solid white;'>
                        <h4 style='margin: 0; color: white;'>{name}</h4>
                        <h2 style='margin: 8px 0; color: white;'>{score}</h2>
                    </div>
                    """, unsafe_allow_html=True)

    def _display_score_breakdown(self, results: Dict) -> None:
        """Display detailed score breakdown and distributions"""
        scoring_breakdown = results.get('scoring_breakdown', {})

        if scoring_breakdown:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🧮 Component Scores**")
                component_scores = scoring_breakdown.get('component_scores', {})
                for component, score in component_scores.items():
                    st.progress(score/5, text=f"{component.title()}: {score}/5.0")

            with col2:
                st.markdown("**⚖️ Applied Weights**")
                weights = scoring_breakdown.get('applied_weights', {})
                for component, weight in weights.items():
                    percentage = weight * 100
                    st.write(f"• **{component.title()}:** {percentage:.1f}%")

            # Historical context if available
            history = self.persistence.get_all_history()
            if len(history) > 1:
                st.markdown("**📈 Historical Context**")

                scores = [h['overall_score'] for h in history]
                current_score = results['overall_score']

                avg_score = np.mean(scores)
                percentile = np.sum(np.array(scores) < current_score) / len(scores) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("vs Average", f"{current_score - avg_score:+.1f}")
                with col2:
                    st.metric("Percentile", f"{percentile:.0f}%")
                with col3:
                    st.metric("Position", f"{np.sum(np.array(scores) < current_score) + 1}/{len(scores)}")

# --------------------------
# ENHANCED SIDEBAR COMPONENTS
# --------------------------
def display_enhanced_sidebar() -> None:
    """Display enhanced sidebar with more features"""
    st.sidebar.title("🎬 Fresh-Tomatoes-AI-Film-Meter")
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.subheader("📍 Navigation")
    
    # Navigation buttons with icons
    if st.sidebar.button("🏠 Dashboard", width='stretch', key="sidebar_dashboard"):
        st.session_state.current_page = "🏠 Dashboard"
        st.rerun()
    
    if st.sidebar.button("📈 Advanced Analytics", width='stretch', key="sidebar_analytics"):
        st.session_state.current_page = "📈 Analytics"
        st.rerun()
    
    if st.sidebar.button("🧠 AI Technology", width='stretch', key="sidebar_ai"):
        st.session_state.current_page = "🧠 AI Technology"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Enhanced Quick Stats
    st.sidebar.subheader("📊 Performance Metrics")
    
    films = list(st.session_state.stored_results.values())
    
    if films:
        scores = [film["analysis_results"]["overall_score"] for film in films]
        cultural_films = sum(1 for film in films 
                           if film["analysis_results"].get('cultural_insights', {}).get('is_culturally_relevant', False))
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.sidebar.metric("Films", len(films))
        
        with col2:
            if scores:
                st.sidebar.metric("Avg Score", f"{np.mean(scores):.1f}")
        
        # Additional metrics in expander
        with st.sidebar.expander("More Stats"):
            st.sidebar.write(f"**Cultural Films:** {cultural_films}")
            st.sidebar.write(f"**Analyses:** {st.session_state.analysis_count}")
            if scores:
                st.sidebar.write(f"**Score Range:** {max(scores) - min(scores):.1f}")
            
            if st.session_state.last_analysis_time:
                last_time = datetime.fromisoformat(st.session_state.last_analysis_time)
                st.sidebar.write(f"**Last:** {last_time.strftime('%H:%M')}")
    
    st.sidebar.markdown("---")
    
    # System Controls
    st.sidebar.subheader("⚙️ System")
    
    if st.sidebar.button("🗑️ Clear All History", key="sidebar_clear_all", type="secondary", width='stretch'):
        persistence = PersistenceManager()
        persistence.clear_history()
        st.sidebar.success("✅ History cleared!")
        st.rerun()
    
    # Version info
    st.sidebar.markdown("---")
    st.sidebar.caption("Fresh-Tomatoes-AI-Film-Meter v2.0")
    st.sidebar.caption("Enhanced Analytics Edition")

# --------------------------
# MAIN ENHANCED APPLICATION
# --------------------------
def main() -> None:
    """Main application entry point"""
    # Display enhanced sidebar
    display_enhanced_sidebar()
    
    # Initialize enhanced components
    analyzer = FilmAnalysisEngine()
    persistence = PersistenceManager()
    film_interface = EnhancedFilmAnalysisInterface(analyzer)
    
    # Determine which page to show
    page = st.session_state.current_page
    
    if page == "🏠 Dashboard":
        film_interface.show_dashboard()
    elif page == "📈 Analytics":
        history_page = EnhancedHistoryAnalyticsPage(persistence)
        history_page.show()
    elif page == "🧠 AI Technology":
        # Create a simple AI Technology page
        st.header("🧠 AI Technology & Roadmap")
        st.markdown("---")
        st.markdown("""
        ## 🚀 Next-Generation Film Analysis AI
        
        **Current Technology Stack:**
        - **VADER Sentiment Analysis**: Emotional tone detection
        - **NLTK**: Natural language processing
        - **Custom Algorithms**: Genre and cultural analysis
        - **Statistical Models**: Comprehensive scoring
        
        **Enhancement Roadmap:**
        1. **Phase 1**: BERT/GPT-4 integration for advanced narrative analysis
        2. **Phase 2**: Multimodal analysis (visual + audio + text)
        3. **Phase 3**: Predictive analytics for festival success
        4. **Phase 4**: Real-time production assistant tools
        
        **New Feature - Video Viewer:**
        - Embedded YouTube player for study group analysis
        - Direct video viewing within analysis interface
        - Study group helper with specific moments to watch
        """)
    else:
        # Enhanced About page
        st.header("🌟 About Fresh-Tomatoes-AI-Film-Meter v2.0")
        
        tab1, tab2, tab3 = st.tabs(["Overview", "Features", "Philosophy"])
        
        with tab1:
            st.markdown("""
            ## 🎬 Fresh-Tomatoes-AI-Film-Meter v2.0 - Advanced Film Analysis
            
            **The next generation** of film analysis technology, combining AI intelligence with 
            cultural awareness and philosophical insight.
            
            ### 🚀 What's New in v2.0
            
            **Enhanced Analytics:**
            - Score trends and evolution tracking
            - Genre distribution analysis
            - Cultural relevance insights
            - Time-based pattern recognition
            
            **New Video Viewer:**
            - Embedded YouTube video player
            - Study group analysis tools
            - Direct video viewing with analysis
            
            **Philosophical Framework:**
            - Cultural memory recognition
            - Narrative as truth-seeking
            - Film as empathy machine
            - Artistic intent analysis
            """)
        
        with tab2:
            st.markdown("""
            ## 🌟 Enhanced Features
            
            **📊 Advanced Analytics Dashboard:**
            - Real-time score distribution tracking
            - Genre performance metrics
            - Cultural relevance scoring
            - Historical trend analysis
            
            **🎥 Video Analysis Tools:**
            - Embedded YouTube viewer
            - Transcript analysis
            - Study group helper
            - Key moment identification
            
            **🎭 Philosophical Insights:**
            - Cultural context understanding
            - Narrative pattern recognition
            - Emotional arc analysis
            - Character development assessment
            """)
        
        with tab3:
            st.markdown("""
            ## 💭 Philosophical Foundation
            
            **Our Approach to Film Analysis:**
            
            **1. Cinema as Cultural Artifact:**
            We view films not just as entertainment, but as **cultural artifacts** that 
            reflect and shape society, memory, and identity.
            
            **2. Narrative as Human Experience:**
            Stories are fundamental to human understanding. We analyze narrative structures 
            as **expressions of human experience** and psychological patterns.
            
            **3. Technology as Cultural Interpreter:**
            AI serves as a **cultural interpreter**, identifying patterns and contexts that 
            might be overlooked in traditional analysis, while respecting artistic intent.
            
            **4. Study Group Integration:**
            The embedded video viewer allows **collaborative analysis** where study groups 
            can watch, pause, and discuss films together with real-time scoring feedback.
            """)

if __name__ == "__main__":
    main()
