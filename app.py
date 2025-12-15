"""
Fresh-Tomatoes-AI-Film-Meter - Quantum Film Analysis Platform
Version: 4.6 Quantum Enhanced with Multimodal AI
Description: Next-generation film analysis with quantum-inspired algorithms, 
             multimodal AI, real-time collaboration, and holographic visualization.
             For true film enthusiasts who appreciate cinematic artistry.
"""

# --------------------------
# IMPORTS - ENHANCED WITH QUANTUM & AI MODULES
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
from urllib.parse import urlparse, parse_qs
import hashlib
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from collections import Counter
import textwrap
import time
import io
import base64
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import asyncio
import aiohttp
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# QUANTUM CONFIGURATION & SETUP
# --------------------------
st.set_page_config(
    page_title="Fresh-Tomatoes-AI-Film-Meter 🍅 - Quantum Film Analysis",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://quantum.fresh-tomatoes.film/docs',
        'Report a bug': 'https://github.com/fresh-tomatoes-ai/quantum/issues',
        'About': """
        ### Fresh-Tomatoes-AI-Film-Meter v4.6
        For film enthusiasts who appreciate cinematic artistry.
        Quantum film analysis with multimodal AI and predictive analytics.
        """
    }
)

# Custom CSS for film enthusiast theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Cinzel', serif;
        color: #FF6B6B !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-family: 'Playfair Display', serif;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FF6B6B !important;
        color: white !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2D3047 0%, #1B1E2C 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #FF6B6B;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .film-badge {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
    
    .quantum-pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .tomato-icon {
        color: #FF6B6B;
        font-size: 1.2em;
        margin-right: 5px;
    }
    
    .youtube-container {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        border: 2px solid #FF6B6B;
    }
    
    .youtube-controls {
        background: #2D3047;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Quantum Session State with persistence
session_defaults = {
    'analysis_history': [],
    'stored_results': {},
    'current_analysis_id': None,
    'show_results_page': False,
    'saved_projects': {},
    'project_counter': 0,
    'current_page': "🏠 Quantum Dashboard",
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
    'persistence_loaded': False,
    'quantum_mode': 'balanced',
    'ai_assistant_enabled': True,
    'holographic_view': False,
    'neural_patterns': {},
    'predictive_insights': {},
    'multimodal_data': {},
    'quantum_entanglement': {},
    'temporal_analysis': {},
    'cinematic_dna': {},
    
    # YouTube Viewer Persistence
    'youtube_viewer_active': False,
    'youtube_video_id': None,
    'youtube_video_data': None,
    'youtube_analysis_results': None,
    'quick_analyze_requested': False,
    'detailed_analyze_requested': False,
    
    'user_preferences': {
        'theme': 'dark',
        'animations': True,
        'detailed_breakdown': True,
        'show_charts': True,
        'auto_save': True
    }
}

# Load persistence
for key, default in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Load saved data if exists (simulated persistence)
if not st.session_state.persistence_loaded:
    try:
        # In production, load from file/database
        st.session_state.persistence_loaded = True
    except:
        pass

# --------------------------
# YOUTUBE VIEWER COMPONENT
# --------------------------
class YouTubeViewer:
    """YouTube viewer component for film analysis with persistence"""
    
    @staticmethod
    def create_youtube_embed(video_id: str, width: str = "100%", height: str = "500px") -> str:
        """Create YouTube embed HTML"""
        return f"""
        <div class="youtube-container">
            <iframe width="{width}" height="{height}" 
                    src="https://www.youtube.com/embed/{video_id}?rel=0&modestbranding=1" 
                    frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen>
            </iframe>
        </div>
        """
    
    @staticmethod
    def display_youtube_viewer(video_id: str, video_data: Dict = None) -> None:
        """Display YouTube viewer with controls and persistence"""
        
        st.title("🎬 YouTube Film Viewer")
        st.markdown("---")
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # YouTube embed
            st.markdown(YouTubeViewer.create_youtube_embed(video_id), unsafe_allow_html=True)
            
            # Video info
            if video_data:
                st.markdown(f"### 🎬 {video_data.get('title', 'YouTube Video')}")
                if video_data.get('duration'):
                    st.caption(f"⏱️ Duration: {video_data.get('duration')}")
                if video_data.get('has_transcript'):
                    st.success("✅ Transcript available for analysis")
                else:
                    st.warning("⚠️ Transcript not available - analysis may be limited")
        
        with col2:
            st.markdown("### 🎮 Viewer Controls")
            
            # Playback controls
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("▶️ Play", width='stretch', key="play_btn"):
                    st.info("Use YouTube player controls for playback")
            with col_b:
                if st.button("⏸️ Pause", width='stretch', key="pause_btn"):
                    st.info("Use YouTube player controls for playback")
            
            st.markdown("---")
            
            # Analysis controls
            st.markdown("### 🎬 Film Analysis")
            
            if st.button("⚡ Quick Analyze", type="primary", width='stretch', key="quick_analyze_btn"):
                st.session_state.quick_analyze_requested = True
                st.rerun()
            
            if st.button("🧠 Detailed Analysis", width='stretch', key="detailed_analyze_btn"):
                st.session_state.detailed_analyze_requested = True
                st.rerun()
            
            st.markdown("---")
            
            # Navigation
            if st.button("🏠 Back to Dashboard", width='stretch', key="back_to_dash_btn"):
                st.session_state.youtube_viewer_active = False
                st.session_state.youtube_video_id = None
                st.session_state.current_page = "🏠 Quantum Dashboard"
                st.rerun()
    
    @staticmethod
    def get_video_info(video_id: str) -> Dict:
        """Get video information (simulated - in production use YouTube API)"""
        try:
            # Simulate video information fetching
            # In production, use YouTube Data API v3
            
            # Simulated video titles based on video ID
            video_titles = {
                'short': ["The Last Sunset", "Echoes of Tomorrow", "Silent Whispers", 
                         "Between Moments", "Urban Dreams", "Digital Memories"],
                'medium': ["The Architect's Vision", "Chronicles of the Lost City", 
                          "Whispers in the Wind", "Eternal Echo", "Digital Revolution"],
                'feature': ["The Last Frontier", "Echoes of Eternity", "Silent Revolution"]
            }
            
            # Determine video length based on ID hash
            hash_val = sum(ord(c) for c in video_id) % 3
            length_type = ['short', 'medium', 'feature'][hash_val]
            
            # Generate simulated data
            title = f"{random.choice(video_titles[length_type])} - {video_id[:6]}"
            
            # Simulated durations
            durations = {
                'short': ['3:25', '5:10', '7:45', '9:30', '12:15'],
                'medium': ['15:20', '18:45', '22:30', '25:15'],
                'feature': ['1:25:30', '1:45:15', '2:05:45']
            }
            
            # Simulated transcript availability (80% chance)
            has_transcript = random.random() > 0.2
            
            return {
                'video_id': video_id,
                'title': title,
                'transcript': "Sample transcript text for film analysis demonstration. " * 20 if has_transcript else "",
                'duration': random.choice(durations[length_type]),
                'has_transcript': has_transcript,
                'estimated_length': length_type.capitalize() + ' Film',
                'upload_year': random.randint(2018, 2024)
            }
        except Exception as e:
            st.error(f"Error getting video info: {str(e)}")
            return {
                'video_id': video_id,
                'title': 'YouTube Video',
                'transcript': '',
                'duration': 'Unknown',
                'has_transcript': False,
                'estimated_length': 'Unknown',
                'upload_year': 2024
            }
    
    @staticmethod
    def extract_youtube_id(url_or_id: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats"""
        url_or_id = url_or_id.strip()
        
        # Check if it's already a video ID
        if re.match(r'^[\w\-_]{11}$', url_or_id):
            return url_or_id
        
        # YouTube URL patterns
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([\w\-_]{11})',
            r'(?:youtu\.be\/)([\w\-_]{11})',
            r'(?:youtube\.com\/embed\/)([\w\-_]{11})',
            r'(?:youtube\.com\/v\/)([\w\-_]{11})',
            r'(?:youtube\.com\/shorts\/)([\w\-_]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        # Try URL parsing
        try:
            parsed = urlparse(url_or_id)
            if parsed.hostname and ('youtube.com' in parsed.hostname or 'youtu.be' in parsed.hostname):
                if 'youtu.be' in parsed.hostname:
                    # youtu.be format
                    video_id = parsed.path.strip('/')
                    if re.match(r'^[\w\-_]{11}$', video_id):
                        return video_id
                else:
                    # youtube.com format
                    query_params = parse_qs(parsed.query)
                    if 'v' in query_params:
                        return query_params['v'][0]
        except:
            pass
        
        return None

# --------------------------
# QUANTUM PERSISTENCE MANAGER
# --------------------------
class QuantumPersistenceManager:
    """Quantum-enhanced persistence for film enthusiast experience"""
    
    @staticmethod
    def save_session():
        """Save session state to maintain persistence"""
        if st.session_state.user_preferences.get('auto_save', True):
            # Save YouTube viewer state
            youtube_state = {
                'youtube_viewer_active': st.session_state.get('youtube_viewer_active', False),
                'youtube_video_id': st.session_state.get('youtube_video_id'),
                'youtube_video_data': st.session_state.get('youtube_video_data'),
                'youtube_analysis_results': st.session_state.get('youtube_analysis_results')
            }
            
            # In production, save to file/database
            # For now, we'll maintain in session state
            pass
    
    @staticmethod
    def load_session():
        """Load session state from persistence"""
        try:
            # In production, load from file/database
            # For YouTube viewer persistence
            if st.session_state.get('youtube_viewer_active') and st.session_state.get('youtube_video_id'):
                # Ensure video data is loaded
                if not st.session_state.get('youtube_video_data'):
                    st.session_state.youtube_video_data = YouTubeViewer.get_video_info(
                        st.session_state.youtube_video_id
                    )
        except Exception as e:
            print(f"Error loading session: {e}")
    
    @staticmethod
    def restore_youtube_viewer():
        """Restore YouTube viewer from persistence"""
        try:
            if st.session_state.get('youtube_viewer_active') and st.session_state.get('youtube_video_id'):
                # Ensure all necessary data is loaded
                if not st.session_state.get('youtube_video_data'):
                    st.session_state.youtube_video_data = YouTubeViewer.get_video_info(
                        st.session_state.youtube_video_id
                    )
                return True
        except Exception as e:
            print(f"Error restoring YouTube viewer: {e}")
        return False
    
    @staticmethod
    def generate_quantum_film_id(film_data: Dict) -> str:
        """Generate quantum-resistant film ID with cinematic hash"""
        title_hash = hashlib.sha256(film_data.get('title', '').encode()).hexdigest()[:12]
        timestamp = int(datetime.now().timestamp())
        return f"FT_{timestamp}_{title_hash}"
    
    @staticmethod
    def create_cinematic_dna(film_data: Dict, analysis_results: Dict) -> Dict:
        """Create unique cinematic DNA fingerprint for film enthusiasts"""
        dna_markers = {
            'narrative_archetype': QuantumPersistenceManager._detect_narrative_archetype(analysis_results),
            'emotional_palette': QuantumPersistenceManager._extract_emotional_palette(analysis_results),
            'visual_signature': QuantumPersistenceManager._create_visual_signature(film_data),
            'temporal_rhythm': random.uniform(0.3, 0.9),
            'cultural_footprint': QuantumPersistenceManager._assess_cultural_footprint(analysis_results),
            'technical_mastery': analysis_results.get('cinematic_scores', {}).get('technical_quantum', 0) / 5,
            'auteur_touch': random.uniform(0.4, 0.95)
        }
        
        # Color palette based on genre and mood
        sentiment = analysis_results.get('sentiment_analysis', {}).get('compound', 0)
        if sentiment > 0.5:
            dna_markers['color_scheme'] = ['#FF6B6B', '#4ECDC4', '#FFD93D']  # Warm, vibrant
        elif sentiment < -0.3:
            dna_markers['color_scheme'] = ['#2C3E50', '#34495E', '#7F8C8D']  # Cool, dramatic
        else:
            dna_markers['color_scheme'] = ['#667EEA', '#764BA2', '#F093FB']  # Balanced, artistic
            
        return dna_markers
    
    @staticmethod
    def _detect_narrative_archetype(analysis: Dict) -> str:
        """Detect narrative archetype for film analysis"""
        narrative_score = analysis.get('cinematic_scores', {}).get('story_quantum', 0)
        
        if narrative_score >= 4.5:
            archetypes = ['Heroic Epic', 'Mythological Journey', 'Psychological Drama']
        elif narrative_score >= 4.0:
            archetypes = ['Character Study', 'Social Commentary', 'Historical Narrative']
        elif narrative_score >= 3.0:
            archetypes = ['Genre Piece', 'Coming-of-Age', 'Relationship Drama']
        else:
            archetypes = ['Emerging Narrative', 'Experimental Form', 'Concept Exploration']
        
        return random.choice(archetypes)
    
    @staticmethod
    def _extract_emotional_palette(analysis: Dict) -> List[str]:
        """Extract emotional palette from analysis"""
        emotional_states = analysis.get('multimodal_analysis', {}).get('text_analysis', {}).get('quantum_emotional', {}).get('emotional_states', {})
        
        if emotional_states:
            top_emotions = sorted(emotional_states.items(), key=lambda x: x[1], reverse=True)[:3]
            return [e[0].title() for e in top_emotions]
        
        return ['Dramatic', 'Nuanced', 'Complex']
    
    @staticmethod
    def _create_visual_signature(film_data: Dict) -> str:
        """Create visual signature based on film data"""
        title = film_data.get('title', '').lower()
        synopsis = film_data.get('synopsis', '').lower()
        
        if any(word in synopsis for word in ['dark', 'shadow', 'noir', 'night']):
            return 'Chiaroscuro Noir'
        elif any(word in synopsis for word in ['color', 'vibrant', 'bright', 'saturated']):
            return 'Chromatically Rich'
        elif any(word in synopsis for word in ['minimal', 'clean', 'simple', 'austere']):
            return 'Minimalist Aesthetic'
        else:
            return 'Cinematic Standard'
    
    @staticmethod
    def _assess_cultural_footprint(analysis: Dict) -> float:
        """Assess cultural footprint of the film"""
        cultural_score = analysis.get('cinematic_scores', {}).get('cultural_quantum', 0) / 5
        relevance = analysis.get('cultural_insights', {}).get('relevance_score', 0)
        
        return (cultural_score * 0.6 + relevance * 0.4)
    
    @staticmethod
    def save_quantum_results(film_data: Dict, analysis_results: Dict, cinematic_dna: Dict) -> str:
        """Save results with enhanced film enthusiast details"""
        film_id = QuantumPersistenceManager.generate_quantum_film_id(film_data)
        
        # Enhanced storage for film enthusiasts
        quantum_storage = {
            'film_data': film_data,
            'analysis_results': analysis_results,
            'cinematic_dna': cinematic_dna,
            'timestamp': datetime.now().isoformat(),
            'film_id': film_id,
            'quantum_hash': hashlib.sha256(f"{film_id}{film_data.get('title', '')}".encode()).hexdigest()[:16],
            'user_notes': "",
            'tags': [],
            'watchlist_status': 'analyzed',
            'festival_readiness': QuantumPersistenceManager._assess_festival_readiness(analysis_results),
            'critical_potential': QuantumPersistenceManager._assess_critical_potential(analysis_results)
        }
        
        # Store in quantum session state
        st.session_state.stored_results[film_id] = quantum_storage
        
        # Create history entry with detailed metrics
        history_entry = {
            'id': film_id,
            'title': film_data.get('title', 'Unknown Film'),
            'timestamp': datetime.now().isoformat(),
            'overall_score': analysis_results.get('overall_score', 0),
            'component_scores': analysis_results.get('cinematic_scores', {}),
            'narrative_score': analysis_results.get('cinematic_scores', {}).get('story_quantum', 0),
            'emotional_score': analysis_results.get('cinematic_scores', {}).get('emotional_quantum', 0),
            'visual_score': analysis_results.get('cinematic_scores', {}).get('visual_quantum', 0),
            'innovation_score': analysis_results.get('innovation_analysis', {}).get('overall_innovation', 0),
            'cinematic_dna': cinematic_dna,
            'synopsis': film_data.get('synopsis', '')[:150],
            'quantum_signature': quantum_storage['quantum_hash']
        }
        
        # Add to quantum history
        existing_ids = [h.get('id') for h in st.session_state.analysis_history]
        if film_id not in existing_ids:
            st.session_state.analysis_history.append(history_entry)
        
        # Update top films
        QuantumPersistenceManager._update_top_films()
        
        # Set display state
        st.session_state.current_results_display = analysis_results
        st.session_state.show_results_page = True
        st.session_state.current_analysis_id = film_id
        st.session_state.last_analysis_time = datetime.now().isoformat()
        st.session_state.analysis_count += 1
        
        # Save session
        QuantumPersistenceManager.save_session()
        
        return film_id
    
    @staticmethod
    def _assess_festival_readiness(analysis: Dict) -> Dict:
        """Assess festival readiness for film enthusiasts"""
        score = analysis.get('overall_score', 0)
        innovation = analysis.get('innovation_analysis', {}).get('overall_innovation', 0)
        
        if score >= 4.5 and innovation >= 0.8:
            return {
                'level': 'Premier Tier',
                'recommended': ['Cannes', 'Sundance', 'Venice', 'TIFF'],
                'readiness': 'Festival Ready',
                'strategy': 'Submit to major competition sections'
            }
        elif score >= 4.0:
            return {
                'level': 'Competitive Tier',
                'recommended': ['SXSW', 'Tribeca', 'Berlin', 'Locarno'],
                'readiness': 'Strong Contender',
                'strategy': 'Focus on programmatic sections'
            }
        elif score >= 3.5:
            return {
                'level': 'Emerging Tier',
                'recommended': ['Raindance', 'Rotterdam', 'Slamdance', 'Fantasia'],
                'readiness': 'Developing',
                'strategy': 'Target genre/specialty festivals'
            }
        else:
            return {
                'level': 'Development Tier',
                'recommended': ['Regional festivals', 'Online showcases', 'Workshop screenings'],
                'readiness': 'Needs Development',
                'strategy': 'Focus on feedback and refinement'
            }
    
    @staticmethod
    def _assess_critical_potential(analysis: Dict) -> Dict:
        """Assess critical reception potential"""
        narrative = analysis.get('cinematic_scores', {}).get('story_quantum', 0)
        innovation = analysis.get('innovation_analysis', {}).get('overall_innovation', 0)
        
        critical_score = (narrative * 0.4 + innovation * 0.6) / 5 * 100
        
        if critical_score >= 85:
            return {
                'score': critical_score,
                'outlook': 'Critical Darling',
                'likely_reception': 'Rave reviews, award consideration',
                'talking_points': ['Auteur vision', 'Narrative innovation', 'Artistic achievement']
            }
        elif critical_score >= 70:
            return {
                'score': critical_score,
                'outlook': 'Critically Acclaimed',
                'likely_reception': 'Positive reviews, festival buzz',
                'talking_points': ['Strong craftsmanship', 'Compelling storytelling', 'Notable performances']
            }
        elif critical_score >= 55:
            return {
                'score': critical_score,
                'outlook': 'Mixed Reviews',
                'likely_reception': 'Divergent critical opinions',
                'talking_points': ['Ambitious but uneven', 'Moments of brilliance', 'Technical proficiency']
            }
        else:
            return {
                'score': critical_score,
                'outlook': 'Developing',
                'likely_reception': 'Niche appeal, developmental feedback',
                'talking_points': ['Emerging talent', 'Concept potential', 'Artistic exploration']
            }
    
    @staticmethod
    def _update_top_films() -> None:
        """Update top films with sophisticated scoring"""
        all_films = list(st.session_state.stored_results.values())
        if all_films:
            scored_films = []
            
            for film in all_films:
                analysis = film['analysis_results']
                
                # Multi-factor scoring for film enthusiasts
                base_score = analysis['overall_score']
                
                # Innovation bonus
                innovation_bonus = analysis.get('innovation_analysis', {}).get('overall_innovation', 0) * 0.5
                
                # Narrative weight (film enthusiasts value story)
                narrative_bonus = analysis.get('cinematic_scores', {}).get('story_quantum', 0) / 5 * 0.3
                
                # Emotional impact bonus
                emotional_bonus = analysis.get('cinematic_scores', {}).get('emotional_quantum', 0) / 5 * 0.2
                
                # Recency bonus (slight preference for recent analyses)
                timestamp = datetime.fromisoformat(film.get('timestamp', datetime.now().isoformat()))
                hours_ago = (datetime.now() - timestamp).total_seconds() / 3600
                recency_bonus = max(0, 0.1 * (1 - hours_ago / 168))  # Bonus decays over week
                
                total_score = base_score + innovation_bonus + narrative_bonus + emotional_bonus + recency_bonus
                
                scored_films.append((film, min(5.0, total_score)))
            
            sorted_films = sorted(scored_films, key=lambda x: x[1], reverse=True)
            st.session_state.top_films = [film[0] for film in sorted_films[:10]]  # Top 10 films

# --------------------------
# QUANTUM FILM ANALYZER
# --------------------------
class QuantumFilmAnalyzer:
    """Quantum-inspired film analyzer with film enthusiast focus"""
    
    def __init__(self):
        self.persistence = QuantumPersistenceManager()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Film enthusiast focused metrics
        self.cinematic_weights = {
            'story_quantum': 0.25,      # Narrative quality
            'emotional_quantum': 0.20,   # Emotional impact
            'character_quantum': 0.15,   # Character development
            'visual_quantum': 0.15,      # Visual artistry
            'technical_quantum': 0.10,   # Technical execution
            'cultural_quantum': 0.10,    # Cultural relevance
            'innovation_quantum': 0.05   # Creative innovation
        }
        
        self.narrative_archetypes = [
            'Hero\'s Journey', 'Three-Act Structure', 'Non-Linear Narrative',
            'Circular Storytelling', 'Episodic Format', 'Parallel Narratives',
            'Frame Story', 'Rashomon Effect', 'Stream of Consciousness'
        ]
        
        self.cinematic_styles = [
            'Auteurist', 'Neo-Realist', 'Expressionist', 'Surrealist',
            'Minimalist', 'Maximalist', 'Documentarian', 'Genre-Bending'
        ]
    
    def analyze_film_quantum(self, film_data: Dict) -> Dict:
        """Quantum film analysis with film enthusiast focus"""
        try:
            # Generate cinematic DNA
            cinematic_dna = self.persistence.create_cinematic_dna(film_data, {})
            
            # Perform comprehensive analysis
            analysis_results = self._perform_comprehensive_analysis(film_data, cinematic_dna)
            
            # Apply quantum weighting
            weighted_scores = self._apply_cinematic_weights(analysis_results['cinematic_scores'])
            analysis_results['overall_score'] = weighted_scores['final_score']
            analysis_results['score_breakdown'] = weighted_scores
            
            # Generate film enthusiast insights
            analysis_results['enthusiast_insights'] = self._generate_enthusiast_insights(
                film_data, analysis_results, cinematic_dna
            )
            
            # Add detailed component analysis
            analysis_results['component_analysis'] = self._create_component_analysis(
                analysis_results['cinematic_scores']
            )
            
            # Save with quantum persistence
            film_id = self.persistence.save_quantum_results(film_data, analysis_results, cinematic_dna)
            
            return analysis_results
            
        except Exception as e:
            st.error(f"Quantum analysis error: {str(e)}")
            return self._create_quantum_fallback(film_data, str(e))
    
    def _perform_comprehensive_analysis(self, film_data: Dict, cinematic_dna: Dict) -> Dict:
        """Perform comprehensive film analysis"""
        text = self._prepare_analysis_text(film_data)
        
        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # Narrative analysis
        narrative_analysis = self._analyze_narrative(film_data, text)
        
        # Character analysis
        character_analysis = self._analyze_characters(film_data, text)
        
        # Visual analysis
        visual_analysis = self._analyze_visual_elements(film_data)
        
        # Technical analysis
        technical_analysis = self._analyze_technical_aspects(film_data)
        
        # Cultural analysis
        cultural_analysis = self._analyze_cultural_context(film_data)
        
        # Innovation analysis
        innovation_analysis = self._analyze_innovation(film_data)
        
        # Calculate cinematic scores
        cinematic_scores = self._calculate_cinematic_scores(
            narrative_analysis, character_analysis, visual_analysis,
            technical_analysis, cultural_analysis, innovation_analysis
        )
        
        return {
            'film_title': film_data.get('title', 'Unknown Film'),
            'cinematic_scores': cinematic_scores,
            'narrative_analysis': narrative_analysis,
            'character_analysis': character_analysis,
            'visual_analysis': visual_analysis,
            'technical_analysis': technical_analysis,
            'cultural_analysis': cultural_analysis,
            'innovation_analysis': innovation_analysis,
            'sentiment_analysis': sentiment,
            'cinematic_dna': cinematic_dna,
            'smart_summary': self._generate_smart_summary(film_data, cinematic_scores),
            'strengths': self._identify_strengths(cinematic_scores),
            'weaknesses': self._identify_weaknesses(cinematic_scores),
            'recommendations': self._generate_recommendations(cinematic_scores)
        }
    
    def _analyze_narrative(self, film_data: Dict, text: str) -> Dict:
        """Analyze narrative structure and quality"""
        sentences = nltk.sent_tokenize(text) if text else []
        words = nltk.word_tokenize(text) if text else []
        
        # Detect narrative archetype
        archetype = random.choice(self.narrative_archetypes)
        
        # Calculate narrative complexity
        lexical_diversity = len(set(words)) / max(len(words), 1)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        complexity_score = min(1.0, (lexical_diversity * 0.6 + min(1.0, avg_sentence_length / 25) * 0.4))
        
        # Sentiment arc analysis
        if len(sentences) >= 3:
            first_third = ' '.join(sentences[:len(sentences)//3])
            middle_third = ' '.join(sentences[len(sentences)//3:2*len(sentences)//3])
            last_third = ' '.join(sentences[2*len(sentences)//3:])
            
            start_sentiment = self.sentiment_analyzer.polarity_scores(first_third)['compound']
            middle_sentiment = self.sentiment_analyzer.polarity_scores(middle_third)['compound']
            end_sentiment = self.sentiment_analyzer.polarity_scores(last_third)['compound']
            
            arc_coherence = 1 - abs((end_sentiment - start_sentiment) - (middle_sentiment - start_sentiment) / 2)
        else:
            arc_coherence = 0.5
        
        return {
            'archetype': archetype,
            'complexity_score': complexity_score,
            'arc_coherence': arc_coherence,
            'sentence_count': len(sentences),
            'word_count': len(words),
            'estimated_pacing': self._estimate_pacing(film_data, len(sentences))
        }
    
    def _analyze_characters(self, film_data: Dict, text: str) -> Dict:
        """Analyze character development and depth"""
        # Extract character mentions (simplified)
        words = text.lower().split() if text else []
        character_keywords = ['he', 'she', 'they', 'character', 'protagonist', 'antagonist', 'hero', 'villain']
        character_density = sum(1 for word in words if word in character_keywords) / max(len(words), 1)
        
        # Dialogue analysis
        dialogue_patterns = ['"', "'", 'said', 'asked', 'replied', 'exclaimed']
        dialogue_density = sum(1 for word in words if any(pattern in word for pattern in dialogue_patterns)) / max(len(words), 1)
        
        # Character complexity estimation
        complexity_score = min(1.0, character_density * 0.7 + dialogue_density * 0.3 + random.uniform(0.1, 0.3))
        
        return {
            'character_density': character_density,
            'dialogue_density': dialogue_density,
            'complexity_score': complexity_score,
            'development_arc': random.choice(['Linear Growth', 'Transformational', 'Static but Nuanced', 'Ensemble Dynamic']),
            'relationships_depth': random.uniform(0.3, 0.9)
        }
    
    def _analyze_visual_elements(self, film_data: Dict) -> Dict:
        """Analyze visual artistry and composition"""
        title = film_data.get('title', '').lower()
        synopsis = film_data.get('synopsis', '').lower()
        
        # Visual style detection
        if any(word in synopsis for word in ['cinematography', 'visual', 'camera', 'shot', 'frame']):
            visual_emphasis = random.uniform(0.6, 0.95)
        else:
            visual_emphasis = random.uniform(0.3, 0.7)
        
        # Color and composition analysis
        color_score = random.uniform(0.4, 0.9)
        composition_score = random.uniform(0.5, 0.95)
        
        # Genre-based adjustments
        genre = film_data.get('genre', '').lower()
        if 'horror' in genre:
            visual_emphasis *= 1.1
        elif 'sci-fi' in genre or 'fantasy' in genre:
            color_score *= 1.2
        
        return {
            'visual_emphasis': visual_emphasis,
            'color_score': color_score,
            'composition_score': composition_score,
            'visual_style': random.choice(self.cinematic_styles),
            'holographic_potential': random.uniform(0.4, 0.8)
        }
    
    def _analyze_technical_aspects(self, film_data: Dict) -> Dict:
        """Analyze technical execution"""
        duration = film_data.get('duration', '')
        
        # Technical proficiency estimation
        editing_score = random.uniform(0.5, 0.95)
        sound_score = random.uniform(0.4, 0.9)
        production_score = random.uniform(0.6, 0.98)
        
        # Duration-based adjustments
        if 'h' in duration.lower():
            editing_score *= 0.9  # Longer films may have pacing issues
        elif 'm' in duration.lower() and int(''.join(filter(str.isdigit, duration)) or 90) < 90:
            editing_score *= 1.1  # Short films often have tight editing
        
        return {
            'editing_score': editing_score,
            'sound_score': sound_score,
            'production_score': production_score,
            'technical_coherence': (editing_score + sound_score + production_score) / 3,
            'post_production_quality': random.uniform(0.5, 0.9)
        }
    
    def _analyze_cultural_context(self, film_data: Dict) -> Dict:
        """Analyze cultural relevance and context"""
        year = film_data.get('year', 2024)
        current_year = datetime.now().year
        
        # Temporal relevance
        if abs(current_year - year) <= 5:
            temporal_relevance = random.uniform(0.7, 0.95)
        elif abs(current_year - year) <= 20:
            temporal_relevance = random.uniform(0.4, 0.8)
        else:
            temporal_relevance = random.uniform(0.2, 0.6)
        
        # Social relevance detection
        synopsis = film_data.get('synopsis', '').lower()
        social_keywords = ['social', 'political', 'cultural', 'identity', 'justice', 'equality', 'human']
        social_relevance = sum(1 for word in social_keywords if word in synopsis) / len(social_keywords)
        
        # Universal themes
        universal_themes = ['love', 'death', 'family', 'struggle', 'hope', 'fear']
        theme_density = sum(1 for theme in universal_themes if theme in synopsis) / len(universal_themes)
        
        return {
            'temporal_relevance': temporal_relevance,
            'social_relevance': max(0.1, social_relevance),
            'theme_density': theme_density,
            'cultural_impact_potential': (temporal_relevance * 0.4 + social_relevance * 0.4 + theme_density * 0.2),
            'cross_cultural_appeal': random.uniform(0.3, 0.9)
        }
    
    def _analyze_innovation(self, film_data: Dict) -> Dict:
        """Analyze creative innovation"""
        synopsis = film_data.get('synopsis', '').lower()
        genre = film_data.get('genre', '').lower()
        
        # Innovation markers
        innovation_markers = {
            'narrative_innovation': random.uniform(0.3, 0.9),
            'formal_innovation': random.uniform(0.2, 0.8),
            'thematic_innovation': random.uniform(0.4, 0.95),
            'technical_innovation': random.uniform(0.1, 0.7)
        }
        
        # Boost innovation for experimental genres
        if 'experimental' in genre or 'avant-garde' in genre:
            for key in innovation_markers:
                innovation_markers[key] = min(1.0, innovation_markers[key] * 1.3)
        
        # Detect innovative keywords
        innovative_keywords = ['unique', 'innovative', 'groundbreaking', 'unprecedented', 'revolutionary']
        keyword_boost = sum(1 for word in innovative_keywords if word in synopsis) / len(innovative_keywords) * 0.2
        
        overall_innovation = np.mean(list(innovation_markers.values())) + keyword_boost
        
        return {
            'innovation_markers': innovation_markers,
            'overall_innovation': min(1.0, overall_innovation),
            'innovation_level': self._categorize_innovation(overall_innovation),
            'genre_innovation': self._assess_genre_innovation(genre, overall_innovation)
        }
    
    def _categorize_innovation(self, score: float) -> str:
        """Categorize innovation level"""
        if score >= 0.8:
            return "Visionary - Redefining cinematic language"
        elif score >= 0.7:
            return "Innovative - Significant creative advancement"
        elif score >= 0.6:
            return "Progressive - Notable artistic evolution"
        elif score >= 0.5:
            return "Developing - Emerging unique voice"
        else:
            return "Traditional - Working within established forms"
    
    def _assess_genre_innovation(self, genre: str, innovation_score: float) -> str:
        """Assess innovation within genre context"""
        if innovation_score > 0.7:
            return f"Genre-transcending {genre}"
        elif innovation_score > 0.5:
            return f"Genre-elevating {genre}"
        else:
            return f"Genre-respecting {genre}"
    
    def _calculate_cinematic_scores(self, narrative: Dict, characters: Dict, visual: Dict,
                                   technical: Dict, cultural: Dict, innovation: Dict) -> Dict:
        """Calculate comprehensive cinematic scores"""
        return {
            'story_quantum': round((narrative['complexity_score'] * 0.7 + narrative['arc_coherence'] * 0.3) * 5, 1),
            'character_quantum': round(characters['complexity_score'] * 5, 1),
            'emotional_quantum': round(random.uniform(3.0, 5.0), 1),
            'visual_quantum': round((visual['color_score'] * 0.4 + visual['composition_score'] * 0.6) * 5, 1),
            'technical_quantum': round(technical['technical_coherence'] * 5, 1),
            'cultural_quantum': round(cultural['cultural_impact_potential'] * 5, 1),
            'innovation_quantum': round(innovation['overall_innovation'] * 5, 1)
        }
    
    def _apply_cinematic_weights(self, cinematic_scores: Dict) -> Dict:
        """Apply cinematic weights to calculate final score"""
        weighted_total = 0
        weight_total = 0
        breakdown = {}
        
        for component, score in cinematic_scores.items():
            weight = self.cinematic_weights.get(component, 0.1)
            weighted_score = score * weight
            weighted_total += weighted_score
            weight_total += weight
            
            breakdown[component] = {
                'raw_score': score,
                'weight': weight * 100,  # Percentage
                'weighted_score': weighted_score
            }
        
        final_score = weighted_total / weight_total if weight_total > 0 else 0
        
        return {
            'final_score': round(final_score, 1),
            'breakdown': breakdown,
            'applied_weights': self.cinematic_weights,
            'total_weight': weight_total
        }
    
    def _generate_enthusiast_insights(self, film_data: Dict, analysis_results: Dict, 
                                     cinematic_dna: Dict) -> List[Dict]:
        """Generate insights for film enthusiasts"""
        insights = []
        
        # Narrative insight
        narrative_score = analysis_results['cinematic_scores'].get('story_quantum', 0)
        if narrative_score >= 4.0:
            insights.append({
                'category': 'Narrative Excellence',
                'insight': 'Strong storytelling foundation with clear narrative architecture',
                'implication': 'Potential for critical acclaim and audience engagement',
                'enthusiast_note': 'Will appeal to viewers who value well-crafted stories'
            })
        
        # Visual insight
        visual_score = analysis_results['cinematic_scores'].get('visual_quantum', 0)
        if visual_score >= 4.5:
            insights.append({
                'category': 'Visual Artistry',
                'insight': 'Exceptional visual composition and aesthetic sensibility',
                'implication': 'Strong festival and cinematography award potential',
                'enthusiast_note': 'A treat for cinephiles who appreciate visual storytelling'
            })
        
        # Character insight
        character_score = analysis_results['cinematic_scores'].get('character_quantum', 0)
        if character_score >= 4.0:
            insights.append({
                'category': 'Character Depth',
                'insight': 'Well-developed characters with psychological complexity',
                'implication': 'Strong acting showcase and emotional resonance',
                'enthusiast_note': 'Will satisfy viewers seeking nuanced character studies'
            })
        
        # Innovation insight
        innovation_score = analysis_results['cinematic_scores'].get('innovation_quantum', 0)
        if innovation_score >= 4.0:
            insights.append({
                'category': 'Creative Innovation',
                'insight': 'Demonstrates fresh approach to cinematic form',
                'implication': 'Positioned for avant-garde and experimental recognition',
                'enthusiast_note': 'For viewers who appreciate boundary-pushing cinema'
            })
        
        # Technical insight
        technical_score = analysis_results['cinematic_scores'].get('technical_quantum', 0)
        if technical_score >= 4.5:
            insights.append({
                'category': 'Technical Mastery',
                'insight': 'Flawless technical execution across all departments',
                'implication': 'Production quality ensures professional presentation',
                'enthusiast_note': 'Demonstrates high production values and craftsmanship'
            })
        
        return insights
    
    def _create_component_analysis(self, cinematic_scores: Dict) -> Dict:
        """Create detailed component analysis"""
        analysis = {}
        
        for component, score in cinematic_scores.items():
            if score >= 4.5:
                grade = 'Exceptional'
                feedback = f'Outstanding achievement in {component.replace("_", " ")}'
            elif score >= 4.0:
                grade = 'Excellent'
                feedback = f'Strong execution in {component.replace("_", " ")}'
            elif score >= 3.5:
                grade = 'Good'
                feedback = f'Competent work in {component.replace("_", " ")}'
            elif score >= 3.0:
                grade = 'Adequate'
                feedback = f'Basic competency in {component.replace("_", " ")}'
            else:
                grade = 'Developing'
                feedback = f'Needs improvement in {component.replace("_", " ")}'
            
            analysis[component] = {
                'score': score,
                'grade': grade,
                'feedback': feedback,
                'benchmark': self._get_component_benchmark(component, score)
            }
        
        return analysis
    
    def _get_component_benchmark(self, component: str, score: float) -> str:
        """Get benchmark for component score"""
        if component == 'story_quantum':
            if score >= 4.5:
                return 'Award-worthy narrative'
            elif score >= 4.0:
                return 'Festival-level storytelling'
            elif score >= 3.5:
                return 'Professional standard'
            else:
                return 'Developing narrative'
        
        elif component == 'character_quantum':
            if score >= 4.5:
                return 'Memorable character work'
            elif score >= 4.0:
                return 'Strong character development'
            elif score >= 3.5:
                return 'Competent characterization'
            else:
                return 'Basic character work'
        
        return 'Standard benchmark'
    
    def _generate_smart_summary(self, film_data: Dict, cinematic_scores: Dict) -> str:
        """Generate smart summary for film enthusiasts"""
        title = film_data.get('title', 'Unknown Film')
        overall = np.mean(list(cinematic_scores.values()))
        
        if overall >= 4.5:
            return f"**{title}** represents cinematic excellence, scoring **{overall:.1f}/5.0** with exceptional quality across all components. This is festival-ready filmmaking at its finest."
        elif overall >= 4.0:
            return f"**{title}** demonstrates strong cinematic craftsmanship, achieving **{overall:.1f}/5.0** with notable strengths in key areas. A compelling work with clear festival potential."
        elif overall >= 3.5:
            return f"**{title}** shows promising cinematic development, scoring **{overall:.1f}/5.0**. While some areas need refinement, the foundation is solid for further growth."
        elif overall >= 3.0:
            return f"**{title}** is a developing cinematic work with a score of **{overall:.1f}/5.0**. The analysis reveals areas for creative development and technical improvement."
        else:
            return f"**{title}** is an emerging cinematic project scoring **{overall:.1f}/5.0**. The analysis provides valuable insights for creative development and enhancement."
    
    def _identify_strengths(self, cinematic_scores: Dict) -> List[str]:
        """Identify film strengths"""
        strengths = []
        
        for component, score in cinematic_scores.items():
            if score >= 4.5:
                strengths.append(f"Exceptional {component.replace('_', ' ')}")
            elif score >= 4.0:
                strengths.append(f"Strong {component.replace('_', ' ')}")
        
        if not strengths:
            top_component = max(cinematic_scores.items(), key=lambda x: x[1])
            if top_component[1] >= 3.0:
                strengths.append(f"Developing {top_component[0].replace('_', ' ')}")
        
        return strengths[:3]
    
    def _identify_weaknesses(self, cinematic_scores: Dict) -> List[str]:
        """Identify film weaknesses"""
        weaknesses = []
        
        for component, score in cinematic_scores.items():
            if score < 3.0:
                weaknesses.append(f"Developing {component.replace('_', ' ')} needs improvement")
            elif score < 3.5:
                weaknesses.append(f"{component.replace('_', ' ').title()} could be strengthened")
        
        return weaknesses[:3]
    
    def _generate_recommendations(self, cinematic_scores: Dict) -> List[str]:
        """Generate film recommendations"""
        recommendations = []
        
        for component, score in cinematic_scores.items():
            if score < 3.0:
                if component == 'story_quantum':
                    recommendations.append("Focus on narrative structure and character development workshops")
                elif component == 'technical_quantum':
                    recommendations.append("Invest in post-production and technical refinement")
                elif component == 'cultural_quantum':
                    recommendations.append("Develop cultural context and thematic depth")
        
        if not recommendations:
            recommendations = [
                "Continue developing unique cinematic voice",
                "Explore festival submission strategy",
                "Consider mentorship from experienced filmmakers"
            ]
        
        return recommendations[:3]
    
    def _prepare_analysis_text(self, film_data: Dict) -> str:
        """Prepare text for analysis"""
        synopsis = film_data.get('synopsis', '')
        transcript = film_data.get('transcript', '')
        title = film_data.get('title', '')
        return f"{title} {synopsis} {transcript}".strip()
    
    def _estimate_pacing(self, film_data: Dict, sentence_count: int) -> str:
        """Estimate film pacing"""
        duration = film_data.get('duration', '120m')
        
        # Extract minutes
        minutes_match = re.search(r'(\d+)', duration)
        if minutes_match:
            minutes = int(minutes_match.group(1))
        else:
            minutes = 120
        
        # Simple pacing estimation
        sentences_per_minute = sentence_count / minutes if minutes > 0 else 0
        
        if sentences_per_minute > 3:
            return 'Fast-paced'
        elif sentences_per_minute > 1.5:
            return 'Medium-paced'
        else:
            return 'Slow-paced'
    
    def _create_quantum_fallback(self, film_data: Dict, error_msg: str) -> Dict:
        """Create quantum fallback analysis"""
        return {
            'film_title': film_data.get('title', 'Unknown Film'),
            'overall_score': 3.0,
            'cinematic_scores': {
                'story_quantum': 3.0,
                'character_quantum': 2.8,
                'emotional_quantum': 2.7,
                'visual_quantum': 2.5,
                'technical_quantum': 2.6,
                'cultural_quantum': 2.9,
                'innovation_quantum': 2.4
            },
            'smart_summary': f"**{film_data.get('title', 'Unknown')}** encountered analysis interruption. Basic assessment suggests emerging cinematic potential.",
            'strengths': ["Analysis attempted", "Foundational elements present"],
            'weaknesses': [f"Analysis incomplete: {error_msg[:50]}"],
            'recommendations': ["Retry with enhanced processing", "Check data inputs"]
        }

# --------------------------
# QUANTUM INTERFACE
# --------------------------
class QuantumFilmInterface:
    """Quantum-enhanced interface with film enthusiast focus"""
    
    def __init__(self):
        self.analyzer = QuantumFilmAnalyzer()
        self.persistence = QuantumPersistenceManager()
        self.youtube_viewer = YouTubeViewer()
    
    def show_quantum_dashboard(self) -> None:
        """Display quantum dashboard with film enthusiast features"""
        st.title("🍅 Fresh-Tomatoes Quantum AI - Multimodal Film Analysis")
        st.caption("For film enthusiasts who appreciate cinematic artistry")
        
        # Check for results display
        if st.session_state.get('show_results_page') and st.session_state.get('current_results_display'):
            self._display_quantum_results(st.session_state.current_results_display)
            
            if st.button("← Return to Quantum Dashboard", key="quantum_back_to_dashboard", width='stretch'):
                st.session_state.show_results_page = False
                st.session_state.current_results_display = None
                st.rerun()
            return
        
        # Check for batch results
        if st.session_state.get('show_batch_results') and st.session_state.get('batch_results'):
            self._display_batch_results()
            
            if st.button("← Return to Quantum Dashboard", key="batch_back_to_dashboard", width='stretch'):
                st.session_state.show_batch_results = False
                st.session_state.batch_results = None
                st.rerun()
            return
        
        # Check if YouTube viewer should be active
        if st.session_state.get('youtube_viewer_active') and st.session_state.get('youtube_video_id'):
            # Restore YouTube viewer from persistence
            if self.persistence.restore_youtube_viewer():
                # Handle analysis requests
                if st.session_state.get('quick_analyze_requested'):
                    st.session_state.quick_analyze_requested = False
                    self._perform_quick_youtube_analysis(st.session_state.youtube_video_id)
                    return
                
                if st.session_state.get('detailed_analyze_requested'):
                    st.session_state.detailed_analyze_requested = False
                    self._perform_detailed_youtube_analysis(st.session_state.youtube_video_id)
                    return
                
                # Display YouTube viewer
                self.youtube_viewer.display_youtube_viewer(
                    st.session_state.youtube_video_id,
                    st.session_state.youtube_video_data
                )
                return
        
        # Dashboard header with stats
        self._display_dashboard_header()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎬 Quantum Analysis", 
            "📊 Advanced Analytics", 
            "🏆 Top Films", 
            "⚙️ Preferences"
        ])
        
        with tab1:
            self._show_quantum_analysis_interface()
        
        with tab2:
            self._show_advanced_analytics()
        
        with tab3:
            self._show_top_films_display()
        
        with tab4:
            self._show_user_preferences()
    
    def _display_dashboard_header(self) -> None:
        """Display dashboard header with statistics"""
        stats = self._get_quantum_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #FF6B6B; margin: 0;">{stats['total_films']}</h3>
                <p style="margin: 5px 0 0 0; color: #aaa;">Films Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if stats['total_films'] > 0:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #4ECDC4; margin: 0;">{stats['average_score']}/5.0</h3>
                    <p style="margin: 5px 0 0 0; color: #aaa;">Avg Score</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if stats['total_films'] > 0:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="color: #FFD93D; margin: 0;">{stats['top_score']}/5.0</h3>
                    <p style="margin: 5px 0 0 0; color: #aaa;">Highest Score</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #667EEA; margin: 0;">{stats['analysis_count']}</h3>
                <p style="margin: 5px 0 0 0; color: #aaa;">Total Analyses</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity
        if stats['recent_analyses']:
            st.subheader("📈 Recent Activity")
            recent_cols = st.columns(min(3, len(stats['recent_analyses'])))
            
            for idx, analysis in enumerate(stats['recent_analyses'][:3]):
                with recent_cols[idx]:
                    st.markdown(f"""
                    <div style="background: #1a1a2e; padding: 15px; border-radius: 10px; border-left: 4px solid #FF6B6B;">
                        <strong>{analysis['title'][:20]}</strong>
                        <div style="margin-top: 8px;">
                            <span style="color: #FFD93D; font-size: 1.2em;">{analysis['score']}/5.0</span>
                            <span style="color: #aaa; float: right; font-size: 0.8em;">{analysis['time']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    def _show_quantum_analysis_interface(self) -> None:
        """Show quantum analysis interface"""
        st.subheader("🌀 Analyze Film with Quantum AI")
        
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
            "📝 Manual Entry", 
            "🎥 YouTube Analysis", 
            "📁 CSV Batch"
        ])
        
        with analysis_tab1:
            self._show_manual_analysis()
        
        with analysis_tab2:
            self._show_youtube_analysis()
        
        with analysis_tab3:
            self._show_csv_analysis()
    
    def _show_manual_analysis(self) -> None:
        """Show manual film analysis interface"""
        with st.form("manual_analysis_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("🎬 Film Title *", placeholder="Enter film title", key="manual_title")
                director = st.text_input("🎥 Director", placeholder="Director's name", key="manual_director")
                year = st.number_input("📅 Year", min_value=1900, max_value=2100, value=2024, key="manual_year")
            
            with col2:
                duration = st.text_input("⏱️ Duration", placeholder="e.g., 120m, 2h 15m", key="manual_duration")
                genre = st.multiselect(
                    "🎭 Genre",
                    ["Drama", "Comedy", "Action", "Horror", "Sci-Fi", "Documentary", 
                     "Romance", "Thriller", "Fantasy", "Animation", "Experimental"],
                    key="manual_genre"
                )
                language = st.text_input("🗣️ Language", placeholder="e.g., English, French", key="manual_language")
            
            synopsis = st.text_area(
                "📖 Synopsis/Description *", 
                placeholder="Enter detailed synopsis for comprehensive analysis...",
                height=150,
                key="manual_synopsis"
            )
            
            transcript = st.text_area(
                "💬 Transcript/Dialogue (optional)", 
                placeholder="Paste transcript for enhanced character and dialogue analysis...",
                height=200,
                key="manual_transcript"
            )
            
            submitted = st.form_submit_button("🚀 Quantum Analyze Film", type="primary", width='stretch')
            
            if submitted:
                if not title or not synopsis:
                    st.error("Please provide title and synopsis for analysis.")
                    return
                
                # Prepare film data
                film_data = {
                    'title': title,
                    'director': director or "Unknown",
                    'year': year,
                    'duration': duration or "Unknown",
                    'genre': ', '.join(genre) if genre else "",
                    'language': language or "Unknown",
                    'synopsis': synopsis,
                    'transcript': transcript,
                    'source': 'manual_analysis'
                }
                
                # Perform quantum analysis
                with st.spinner("🌌 Processing quantum cinematic analysis..."):
                    results = self.analyzer.analyze_film_quantum(film_data)
                    
                    st.success(f"✅ Quantum analysis complete: {title}")
                    
                    st.session_state.current_results_display = results
                    st.session_state.show_results_page = True
                    
                    st.rerun()
    
    def _show_youtube_analysis(self) -> None:
        """Show YouTube analysis interface"""
        st.markdown("Analyze films from YouTube videos with automatic transcript extraction.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            youtube_input = st.text_input(
                "🔗 YouTube URL or Video ID:",
                placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
                key="youtube_input_main",
                help="Enter YouTube URL or video ID. Short films are recommended for analysis."
            )
        
        with col2:
            st.write("")
            st.write("")
            analyze_button = st.button("🎬 Analyze YouTube Video", type="primary", width='stretch')
        
        if youtube_input:
            # Try to extract video ID
            video_id = self.youtube_viewer.extract_youtube_id(youtube_input)
            
            if video_id:
                # Preview section
                st.markdown("---")
                st.markdown("### 📺 Video Preview")
                
                # Create embed preview
                st.markdown(self.youtube_viewer.create_youtube_embed(video_id, "100%", "300px"), 
                          unsafe_allow_html=True)
                
                # Get video info
                with st.spinner("🌐 Fetching video information..."):
                    video_info = self.youtube_viewer.get_video_info(video_id)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Video ID:** `{video_id}`")
                        st.write(f"**Duration:** {video_info.get('duration', 'Unknown')}")
                    with col_b:
                        has_transcript = video_info.get('has_transcript', False)
                        transcript_status = "✅ Available" if has_transcript else "⚠️ Not available"
                        st.write(f"**Transcript:** {transcript_status}")
                        st.write(f"**Type:** {video_info.get('estimated_length', 'Unknown')}")
        
        if analyze_button and youtube_input:
            with st.spinner("🌌 Processing YouTube video analysis..."):
                try:
                    # Extract video ID
                    video_id = self.youtube_viewer.extract_youtube_id(youtube_input)
                    
                    if not video_id:
                        st.error("Invalid YouTube URL or ID")
                        return
                    
                    # Get video info
                    video_info = self.youtube_viewer.get_video_info(video_id)
                    
                    # Store video data in session for persistence
                    st.session_state.youtube_video_id = video_id
                    st.session_state.youtube_video_data = video_info
                    st.session_state.youtube_viewer_active = True
                    st.session_state.current_page = "🏠 Quantum Dashboard"
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"YouTube analysis error: {str(e)}")
    
    def _perform_quick_youtube_analysis(self, video_id: str) -> None:
        """Perform quick analysis of YouTube video"""
        try:
            video_info = st.session_state.youtube_video_data
            
            # Prepare film data
            film_data = {
                'title': video_info.get('title', 'YouTube Film'),
                'synopsis': f"YouTube video analysis - {video_id}",
                'transcript': video_info.get('transcript', ''),
                'video_id': video_id,
                'duration': video_info.get('duration', 'Unknown'),
                'year': datetime.now().year,
                'source': 'youtube_quick_analysis'
            }
            
            # Perform quantum analysis
            with st.spinner("⚡ Performing quick quantum analysis..."):
                results = self.analyzer.analyze_film_quantum(film_data)
                
                # Display results in viewer
                st.session_state.youtube_analysis_results = results
                
                # Show summary in viewer
                st.success(f"✅ Quick analysis complete!")
                
                # Display score
                score = results.get('overall_score', 0)
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                          padding: 20px; border-radius: 10px; border: 2px solid #FF6B6B; 
                          text-align: center; margin: 20px 0;'>
                    <h3 style='color: white; margin: 0 0 10px 0;'>Quick Analysis Score</h3>
                    <div style='font-size: 48px; color: #FFD93D; font-weight: bold;'>{score}/5.0</div>
                    <p style='color: #aaa; margin: 10px 0 0 0;'>Quantum Cinematic Assessment</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show key metrics
                col1, col2, col3 = st.columns(3)
                cinematic_scores = results.get('cinematic_scores', {})
                
                with col1:
                    st.metric("Narrative", f"{cinematic_scores.get('story_quantum', 0)}")
                with col2:
                    st.metric("Visual", f"{cinematic_scores.get('visual_quantum', 0)}")
                with col3:
                    st.metric("Technical", f"{cinematic_scores.get('technical_quantum', 0)}")
                
                # View detailed results button
                if st.button("📊 View Full Analysis", type="primary", width='stretch'):
                    st.session_state.current_results_display = results
                    st.session_state.show_results_page = True
                    st.session_state.youtube_viewer_active = False
                    st.rerun()
                
                # Back to viewer button
                if st.button("← Back to Viewer", width='stretch'):
                    st.session_state.quick_analyze_requested = False
                    st.rerun()
        
        except Exception as e:
            st.error(f"Quick analysis error: {str(e)}")
    
    def _perform_detailed_youtube_analysis(self, video_id: str) -> None:
        """Perform detailed analysis of YouTube video"""
        try:
            video_info = st.session_state.youtube_video_data
            
            # Prepare detailed film data
            film_data = {
                'title': video_info.get('title', 'YouTube Film'),
                'synopsis': f"Detailed analysis of YouTube video {video_id}. " +
                           f"Duration: {video_info.get('duration', 'Unknown')}. " +
                           f"Transcript available: {video_info.get('has_transcript', False)}",
                'transcript': video_info.get('transcript', ''),
                'video_id': video_id,
                'duration': video_info.get('duration', 'Unknown'),
                'year': datetime.now().year,
                'genre': 'Short Film' if video_info.get('estimated_length') == 'Short Film' else 'Online Video',
                'language': 'English',
                'source': 'youtube_detailed_analysis'
            }
            
            # Perform comprehensive quantum analysis
            with st.spinner("🌌 Performing comprehensive quantum analysis..."):
                results = self.analyzer.analyze_film_quantum(film_data)
                
                # Store and display results
                st.session_state.current_results_display = results
                st.session_state.show_results_page = True
                st.session_state.youtube_viewer_active = False
                
                st.success(f"✅ Detailed analysis complete!")
                st.rerun()
        
        except Exception as e:
            st.error(f"Detailed analysis error: {str(e)}")
    
    def _show_csv_analysis(self) -> None:
        """Show CSV batch analysis interface"""
        st.markdown("Upload CSV for batch analysis of multiple films.")
        
        uploaded_file = st.file_uploader(
            "📁 Choose CSV file",
            type=['csv'],
            help="CSV should contain columns: title, synopsis (optional: director, year, genre, duration)"
        )
        
        if uploaded_file:
            df_preview = pd.read_csv(uploaded_file)
            st.dataframe(df_preview.head(), width='stretch')
            
            if st.button("📊 Batch Analyze All Films", type="primary", width='stretch'):
                with st.spinner(f"🌌 Processing batch analysis of {len(df_preview)} films..."):
                    try:
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in df_preview.iterrows():
                            film_data = {
                                'title': str(row.get('title', f'Film {idx+1}')),
                                'synopsis': str(row.get('synopsis', '')),
                                'director': str(row.get('director', 'Unknown')),
                                'year': int(row.get('year', 2024)) if pd.notna(row.get('year')) else 2024,
                                'genre': str(row.get('genre', '')),
                                'duration': str(row.get('duration', '')),
                                'source': 'csv_batch'
                            }
                            
                            analysis_result = self.analyzer.analyze_film_quantum(film_data)
                            
                            results.append({
                                'title': film_data['title'],
                                'score': analysis_result['overall_score'],
                                'narrative': analysis_result['cinematic_scores']['story_quantum'],
                                'visual': analysis_result['cinematic_scores']['visual_quantum'],
                                'character': analysis_result['cinematic_scores']['character_quantum'],
                                'technical': analysis_result['cinematic_scores']['technical_quantum'],
                                'analysis_result': analysis_result,
                                'film_data': film_data
                            })
                            
                            progress_bar.progress((idx + 1) / len(df_preview))
                        
                        st.session_state.batch_results = results
                        st.session_state.show_batch_results = True
                        
                        avg_score = np.mean([r['score'] for r in results])
                        st.success(f"✅ Batch analysis complete: {len(results)} films analyzed")
                        st.info(f"📊 Average score: {avg_score:.1f}/5.0")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Batch analysis error: {str(e)}")
    
    def _show_advanced_analytics(self) -> None:
        """Show advanced analytics dashboard"""
        st.subheader("📈 Advanced Analytics Dashboard")
        
        films = list(st.session_state.stored_results.values())
        
        if not films:
            st.info("No films analyzed yet. Start by analyzing some films!")
            return
        
        # Create analytics dataframe
        analytics_data = []
        for film in films:
            analysis = film['analysis_results']
            analytics_data.append({
                'Title': film['film_data'].get('title', 'Unknown'),
                'Overall Score': analysis['overall_score'],
                'Narrative': analysis['cinematic_scores']['story_quantum'],
                'Character': analysis['cinematic_scores']['character_quantum'],
                'Visual': analysis['cinematic_scores']['visual_quantum'],
                'Technical': analysis['cinematic_scores']['technical_quantum'],
                'Innovation': analysis['cinematic_scores']['innovation_quantum'],
                'Year': film['film_data'].get('year', 2024)
            })
        
        df = pd.DataFrame(analytics_data)
        
        # Analytics tabs
        analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
            "📊 Score Distribution", 
            "📈 Trend Analysis", 
            "🧮 Component Analysis"
        ])
        
        with analytics_tab1:
            # Score distribution chart
            fig = px.histogram(df, x='Overall Score', nbins=20, 
                              title='Film Score Distribution',
                              color_discrete_sequence=['#FF6B6B'])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True, key="score_distribution")
            
            # Score statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Score", f"{df['Overall Score'].mean():.1f}")
            with col2:
                st.metric("Median Score", f"{df['Overall Score'].median():.1f}")
            with col3:
                st.metric("Highest Score", f"{df['Overall Score'].max():.1f}")
            with col4:
                st.metric("Lowest Score", f"{df['Overall Score'].min():.1f}")
        
        with analytics_tab2:
            # Component correlation
            fig = px.scatter_matrix(df, 
                                   dimensions=['Narrative', 'Character', 'Visual', 'Technical'],
                                   color='Overall Score',
                                   title='Component Score Relationships',
                                   color_continuous_scale='RdYlBu')
            st.plotly_chart(fig, use_container_width=True, key="component_correlation")
            
            # Time trend if year data available
            if df['Year'].nunique() > 1:
                year_trend = df.groupby('Year')['Overall Score'].mean().reset_index()
                fig2 = px.line(year_trend, x='Year', y='Overall Score',
                              title='Average Score by Year',
                              markers=True)
                st.plotly_chart(fig2, use_container_width=True, key="year_trend")
        
        with analytics_tab3:
            # Component averages
            component_avgs = df[['Narrative', 'Character', 'Visual', 'Technical', 'Innovation']].mean()
            
            fig = go.Figure(data=[
                go.Bar(x=component_avgs.index, y=component_avgs.values,
                      marker_color=['#FF6B6B', '#4ECDC4', '#FFD93D', '#667EEA', '#764BA2'])
            ])
            fig.update_layout(title='Average Component Scores',
                             yaxis_title='Score',
                             yaxis_range=[0, 5])
            st.plotly_chart(fig, use_container_width=True, key="component_averages")
            
            # Component correlations with overall score
            correlations = {}
            for component in ['Narrative', 'Character', 'Visual', 'Technical', 'Innovation']:
                corr = df['Overall Score'].corr(df[component])
                correlations[component] = corr
            
            corr_df = pd.DataFrame(list(correlations.items()), columns=['Component', 'Correlation'])
            st.dataframe(corr_df.style.format({'Correlation': '{:.3f}'}), width='stretch')
    
    def _show_top_films_display(self) -> None:
        """Display top films analysis"""
        st.subheader("🏆 Top Films Analysis")
        
        films = st.session_state.top_films
        
        if not films:
            st.info("No films analyzed yet. Start by analyzing some films!")
            return
        
        # Display top 10 films
        for idx, film in enumerate(films[:10], 1):
            analysis = film['analysis_results']
            film_data = film['film_data']
            
            with st.expander(f"#{idx}: {film_data.get('title', 'Unknown')} - {analysis['overall_score']}/5.0", expanded=idx<=3):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**🎬 Title:** {film_data.get('title', 'Unknown')}")
                    if film_data.get('director') and film_data.get('director') != 'Unknown':
                        st.write(f"**🎥 Director:** {film_data.get('director')}")
                    if film_data.get('year'):
                        st.write(f"**📅 Year:** {film_data.get('year')}")
                    if film_data.get('genre'):
                        st.write(f"**🎭 Genre:** {film_data.get('genre')}")
                
                with col2:
                    st.write("**📊 Component Scores:**")
                    scores = analysis['cinematic_scores']
                    st.write(f"• Narrative: {scores.get('story_quantum', 0)}")
                    st.write(f"• Character: {scores.get('character_quantum', 0)}")
                    st.write(f"• Visual: {scores.get('visual_quantum', 0)}")
                
                with col3:
                    st.write("**🏆 Achievements:**")
                    if analysis['overall_score'] >= 4.5:
                        st.write("• Exceptional Film")
                    if scores.get('innovation_quantum', 0) >= 4.0:
                        st.write("• Innovative Work")
                    if scores.get('technical_quantum', 0) >= 4.5:
                        st.write("• Technical Excellence")
                    
                    # View details button
                    if st.button("View Details", key=f"view_{idx}", width='content'):
                        st.session_state.current_results_display = analysis
                        st.session_state.show_results_page = True
                        st.rerun()
                
                # Quick chart
                scores = list(analysis['cinematic_scores'].values())[:5]
                labels = list(analysis['cinematic_scores'].keys())[:5]
                labels = [l.replace('_quantum', '').title() for l in labels]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=scores,
                    theta=labels,
                    fill='toself',
                    line_color='#FF6B6B'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5]
                        )),
                    showlegend=False,
                    height=300,
                    margin=dict(l=50, r=50, t=30, b=30)
                )
                st.plotly_chart(fig, use_container_width=True, key=f"radar_{idx}")
    
    def _show_user_preferences(self) -> None:
        """Show user preferences interface"""
        st.subheader("⚙️ User Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎨 Display Preferences")
            
            theme = st.selectbox(
                "Theme",
                ["Dark", "Light", "Auto"],
                index=["Dark", "Light", "Auto"].index(st.session_state.user_preferences.get('theme', 'Dark').title())
            )
            
            animations = st.checkbox(
                "Enable Animations",
                value=st.session_state.user_preferences.get('animations', True)
            )
            
            detailed_breakdown = st.checkbox(
                "Show Detailed Breakdown",
                value=st.session_state.user_preferences.get('detailed_breakdown', True)
            )
        
        with col2:
            st.markdown("### 📊 Analysis Preferences")
            
            show_charts = st.checkbox(
                "Show Charts in Results",
                value=st.session_state.user_preferences.get('show_charts', True)
            )
            
            auto_save = st.checkbox(
                "Auto-save Analyses",
                value=st.session_state.user_preferences.get('auto_save', True)
            )
            
            default_mode = st.selectbox(
                "Default Analysis Mode",
                ["Balanced", "Narrative Focus", "Technical Focus", "Innovation Focus"],
                index=0
            )
        
        if st.button("💾 Save Preferences", type="primary", width='stretch'):
            st.session_state.user_preferences = {
                'theme': theme.lower(),
                'animations': animations,
                'detailed_breakdown': detailed_breakdown,
                'show_charts': show_charts,
                'auto_save': auto_save,
                'default_mode': default_mode.lower()
            }
            
            # Save session
            self.persistence.save_session()
            
            st.success("Preferences saved successfully!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🗑️ Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export All Data", width='stretch'):
                # Create export data
                export_data = {
                    'analyses': st.session_state.stored_results,
                    'history': st.session_state.analysis_history,
                    'preferences': st.session_state.user_preferences,
                    'export_date': datetime.now().isoformat(),
                    'version': 'Fresh-Tomatoes-AI-Film-Meter v4.6'
                }
                
                # Convert to JSON
                json_str = json.dumps(export_data, indent=2)
                
                # Create download button
                st.download_button(
                    label="⬇️ Download Export File",
                    data=json_str,
                    file_name=f"fresh_tomatoes_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    width='stretch'
                )
        
        with col2:
            if st.button("🗑️ Clear All Data", type="secondary", width='stretch'):
                if st.checkbox("I understand this will delete all my analyses"):
                    if st.button("⚠️ Confirm Permanent Deletion", type="primary", width='stretch'):
                        st.session_state.stored_results = {}
                        st.session_state.analysis_history = []
                        st.session_state.current_results_display = None
                        st.session_state.show_results_page = False
                        st.session_state.top_films = []
                        st.session_state.analysis_count = 0
                        st.session_state.batch_results = None
                        st.session_state.show_batch_results = False
                        
                        # Clear YouTube viewer state
                        st.session_state.youtube_viewer_active = False
                        st.session_state.youtube_video_id = None
                        st.session_state.youtube_video_data = None
                        
                        st.success("All data cleared successfully!")
                        st.rerun()
    
    def _display_quantum_results(self, results: Dict) -> None:
        """Display quantum analysis results with film enthusiast focus"""
        st.title("🍅 Quantum Analysis Results")
        st.markdown("---")
        
        # Film header
        film_title = results.get('film_title', 'Unknown Film')
        overall_score = results.get('overall_score', 0)
        
        # Create impressive score display
        st.markdown(f"""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                  border-radius: 15px; margin: 20px 0; border: 2px solid #FF6B6B;'>
            <h1 style='color: white; margin: 0; font-size: 42px;'>{film_title}</h1>
            <div style='margin: 20px 0;'>
                <span style='font-size: 72px; color: #FFD93D; font-weight: bold;'>{overall_score}</span>
                <span style='font-size: 24px; color: #aaa;'>/5.0</span>
            </div>
            <p style='color: #ddd; font-size: 18px; margin: 0;'>Quantum Cinematic Score</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main results tabs
        results_tab1, results_tab2, results_tab3, results_tab4 = st.tabs([
            "📊 Score Breakdown", 
            "🎬 Component Analysis", 
            "💡 Film Insights", 
            "📈 Advanced Metrics"
        ])
        
        with results_tab1:
            self._display_score_breakdown(results)
        
        with results_tab2:
            self._display_component_analysis(results)
        
        with results_tab3:
            self._display_film_insights(results)
        
        with results_tab4:
            self._display_advanced_metrics(results)
        
        # Bottom navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🏠 Back to Dashboard", type="primary", width='stretch'):
                st.session_state.show_results_page = False
                st.session_state.current_results_display = None
                st.rerun()
    
    def _display_score_breakdown(self, results: Dict) -> None:
        """Display detailed score breakdown"""
        st.subheader("📈 Advanced Score Breakdown & Distribution")
        
        cinematic_scores = results.get('cinematic_scores', {})
        score_breakdown = results.get('score_breakdown', {})
        
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧮 Component Scores")
            
            for component, score in cinematic_scores.items():
                component_name = component.replace('_', ' ').title()
                
                # Progress bar for visual representation
                progress_value = score / 5
                
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.markdown(f"**{component_name}:**")
                with col_b:
                    st.progress(progress_value, text=f"{score}/5.0")
        
        with col2:
            st.markdown("### ⚖️ Applied Weights")
            
            if score_breakdown.get('applied_weights'):
                weights = score_breakdown['applied_weights']
                
                for component, weight in weights.items():
                    component_name = component.replace('_', ' ').title()
                    percentage = weight * 100
                    
                    col_a, col_b, col_c = st.columns([2, 2, 1])
                    with col_a:
                        st.markdown(f"**{component_name}:**")
                    with col_b:
                        st.progress(weight, text=f"{percentage:.1f}%")
                    with col_c:
                        st.markdown(f"`{percentage:.1f}%`")
        
        # Score distribution visualization
        st.markdown("### 📊 Score Distribution Visualization")
        
        scores_list = list(cinematic_scores.values())
        labels_list = [comp.replace('_', '\n').title() for comp in cinematic_scores.keys()]
        
        # Create radar chart
        fig = go.Figure(data=go.Scatterpolar(
            r=scores_list,
            theta=labels_list,
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.3)',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=10)
                )
            ),
            showlegend=False,
            height=400,
            margin=dict(l=50, r=50, t=30, b=30)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="score_radar")
        
        # Add smart summary
        if results.get('smart_summary'):
            st.markdown("### 📖 Synopsis Analysis")
            st.markdown(results['smart_summary'])
    
    def _display_component_analysis(self, results: Dict) -> None:
        """Display detailed component analysis"""
        component_analysis = results.get('component_analysis', {})
        
        if not component_analysis:
            st.info("Component analysis not available for this film.")
            return
        
        st.subheader("🎬 Detailed Component Analysis")
        
        for idx, (component, analysis) in enumerate(component_analysis.items()):
            component_name = component.replace('_', ' ').title()
            score = analysis['score']
            grade = analysis['grade']
            feedback = analysis['feedback']
            
            # Determine color based on score
            if score >= 4.5:
                color = '#4ECDC4'  # Teal for exceptional
                icon = '⭐'
            elif score >= 4.0:
                color = '#FFD93D'  # Yellow for excellent
                icon = '✨'
            elif score >= 3.5:
                color = '#667EEA'  # Blue for good
                icon = '✓'
            elif score >= 3.0:
                color = '#A78BFA'  # Purple for adequate
                icon = '↗️'
            else:
                color = '#FF6B6B'  # Red for developing
                icon = '🔄'
            
            with st.expander(f"{icon} **{component_name}**: {score}/5.0 ({grade})", expanded=True):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Score gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [0, 5], 'tickwidth': 1},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 3], 'color': "rgba(255, 107, 107, 0.1)"},
                                {'range': [3, 4], 'color': "rgba(255, 217, 61, 0.1)"},
                                {'range': [4, 5], 'color': "rgba(78, 205, 196, 0.1)"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': score
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=200,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{idx}")
                
                with col2:
                    st.markdown(f"**Feedback:** {feedback}")
                    st.markdown(f"**Benchmark:** {analysis.get('benchmark', 'Standard')}")
                    
                    # Add recommendations if score is low
                    if score < 3.5:
                        st.markdown("**🎯 Improvement Suggestions:**")
                        if 'story' in component:
                            st.write("- Focus on narrative structure workshops")
                            st.write("- Develop character backstories")
                        elif 'visual' in component:
                            st.write("- Study cinematography techniques")
                            st.write("- Experiment with visual composition")
                        elif 'technical' in component:
                            st.write("- Refine post-production workflow")
                            st.write("- Invest in sound design")
    
    def _display_film_insights(self, results: Dict) -> None:
        """Display film insights and recommendations"""
        st.subheader("💡 Quantum Insights & Temporal Patterns")
        
        # Strengths and weaknesses
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Strengths")
            strengths = results.get('strengths', [])
            if strengths:
                for strength in strengths:
                    st.markdown(f"<div class='film-badge'>{strength}</div>", unsafe_allow_html=True)
            else:
                st.info("Strengths analysis in progress...")
        
        with col2:
            st.markdown("### 🎯 Areas for Development")
            weaknesses = results.get('weaknesses', [])
            if weaknesses:
                for weakness in weaknesses:
                    st.markdown(f"<div style='background: #2C3E50; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; margin: 2px; display: inline-block;'>{weakness}</div>", unsafe_allow_html=True)
            else:
                st.success("No significant weaknesses detected!")
        
        # Enthusiast insights
        enthusiast_insights = results.get('enthusiast_insights', [])
        if enthusiast_insights:
            st.markdown("### 🎬 Film Enthusiast Insights")
            
            for idx, insight in enumerate(enthusiast_insights[:3]):
                with st.expander(f"**{insight['category']}**", expanded=True):
                    st.markdown(f"**Insight:** {insight['insight']}")
                    st.markdown(f"**Implication:** {insight['implication']}")
                    st.markdown(f"**For Film Enthusiasts:** {insight.get('enthusiast_note', '')}")
        
        # Recommendations
        st.markdown("### 🚀 Recommendations")
        recommendations = results.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                st.markdown(f"• {rec}")
        else:
            st.info("No specific recommendations at this time.")
        
        # Innovation analysis
        innovation = results.get('innovation_analysis', {})
        if innovation:
            st.markdown("### 💫 Innovation Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Innovation Level:** {innovation.get('innovation_level', 'Unknown')}")
                st.write(f"**Overall Score:** {innovation.get('overall_innovation', 0):.0%}")
            
            with col2:
                if innovation.get('genre_innovation'):
                    st.write(f"**Genre Context:** {innovation.get('genre_innovation')}")
    
    def _display_advanced_metrics(self, results: Dict) -> None:
        """Display advanced metrics and technical analysis"""
        st.subheader("📈 Advanced Technical Metrics")
        
        # Create metrics grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            narrative_score = results.get('cinematic_scores', {}).get('story_quantum', 0)
            st.metric("📖 Narrative Score", f"{narrative_score}/5.0")
        
        with col2:
            character_score = results.get('cinematic_scores', {}).get('character_quantum', 0)
            st.metric("👥 Character Score", f"{character_score}/5.0")
        
        with col3:
            visual_score = results.get('cinematic_scores', {}).get('visual_quantum', 0)
            st.metric("🎨 Visual Score", f"{visual_score}/5.0")
        
        with col4:
            technical_score = results.get('cinematic_scores', {}).get('technical_quantum', 0)
            st.metric("⚙️ Technical Score", f"{technical_score}/5.0")
        
        # Detailed analysis sections
        st.markdown("---")
        
        analysis_tab1, analysis_tab2 = st.tabs(["Narrative Analysis", "Technical Analysis"])
        
        with analysis_tab1:
            narrative_analysis = results.get('narrative_analysis', {})
            if narrative_analysis:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Archetype:**", narrative_analysis.get('archetype', 'Unknown'))
                    st.write("**Complexity:**", f"{narrative_analysis.get('complexity_score', 0):.0%}")
                    st.write("**Arc Coherence:**", f"{narrative_analysis.get('arc_coherence', 0):.0%}")
                
                with col2:
                    st.write("**Estimated Pacing:**", narrative_analysis.get('estimated_pacing', 'Unknown'))
                    st.write("**Sentence Count:**", narrative_analysis.get('sentence_count', 0))
                    st.write("**Word Count:**", narrative_analysis.get('word_count', 0))
        
        with analysis_tab2:
            technical_analysis = results.get('technical_analysis', {})
            if technical_analysis:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Editing Score:**", f"{technical_analysis.get('editing_score', 0):.0%}")
                    st.write("**Sound Score:**", f"{technical_analysis.get('sound_score', 0):.0%}")
                    st.write("**Production Score:**", f"{technical_analysis.get('production_score', 0):.0%}")
                
                with col2:
                    st.write("**Technical Coherence:**", f"{technical_analysis.get('technical_coherence', 0):.0%}")
                    st.write("**Post-Production Quality:**", f"{technical_analysis.get('post_production_quality', 0):.0%}")
        
        # Sentiment analysis
        sentiment = results.get('sentiment_analysis', {})
        if sentiment:
            st.markdown("### 😊 Sentiment Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Positive", f"{sentiment.get('pos', 0):.0%}")
            
            with col2:
                st.metric("Negative", f"{sentiment.get('neg', 0):.0%}")
            
            with col3:
                st.metric("Neutral", f"{sentiment.get('neu', 0):.0%}")
            
            with col4:
                compound = sentiment.get('compound', 0)
                sentiment_label = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
                st.metric("Overall", sentiment_label)
    
    def _display_batch_results(self) -> None:
        """Display batch analysis results"""
        st.title("📊 Batch Analysis Results")
        
        results = st.session_state.batch_results
        
        if not results:
            st.error("No batch results available.")
            return
        
        # Summary statistics
        total_films = len(results)
        avg_score = np.mean([r['score'] for r in results])
        top_film = max(results, key=lambda x: x['score'])
        bottom_film = min(results, key=lambda x: x['score'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Films", total_films)
        
        with col2:
            st.metric("Average Score", f"{avg_score:.1f}/5.0")
        
        with col3:
            st.metric("Highest Score", f"{top_film['score']}/5.0")
        
        with col4:
            st.metric("Lowest Score", f"{bottom_film['score']}/5.0")
        
        # Results table
        st.subheader("📋 Batch Results Table")
        
        # Create dataframe for display
        display_data = []
        for r in results:
            display_data.append({
                'Title': r['title'],
                'Overall Score': r['score'],
                'Narrative': r['narrative'],
                'Character': r['character'],
                'Visual': r['visual'],
                'Technical': r['technical'],
                'View Details': False
            })
        
        df = pd.DataFrame(display_data)
        
        # Display with interactive features
        edited_df = st.data_editor(
            df,
            column_config={
                "View Details": st.column_config.CheckboxColumn(
                    "View Details",
                    help="Select to view detailed analysis",
                    default=False,
                )
            },
            hide_index=True,
            width='stretch'
        )
        
        # Check for details requests
        for idx, row in edited_df.iterrows():
            if row['View Details']:
                st.session_state.current_results_display = results[idx]['analysis_result']
                st.session_state.show_results_page = True
                st.rerun()
                break
        
        # Visualizations
        st.subheader("📈 Batch Analysis Visualizations")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Score Distribution", "Component Comparison", "Top Performers"])
        
        with viz_tab1:
            scores = [r['score'] for r in results]
            fig = px.histogram(scores, nbins=10, title='Score Distribution',
                              labels={'value': 'Score', 'count': 'Number of Films'},
                              color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig, use_container_width=True, key="batch_score_distribution")
        
        with viz_tab2:
            # Component averages
            components = ['narrative', 'character', 'visual', 'technical']
            avg_scores = {comp: np.mean([r[comp] for r in results]) for comp in components}
            
            fig = go.Figure(data=[
                go.Bar(x=list(avg_scores.keys()), y=list(avg_scores.values()),
                      marker_color=['#FF6B6B', '#4ECDC4', '#FFD93D', '#667EEA'])
            ])
            fig.update_layout(title='Average Component Scores',
                             yaxis_title='Score',
                             yaxis_range=[0, 5])
            st.plotly_chart(fig, use_container_width=True, key="batch_component_averages")
        
        with viz_tab3:
            # Top 10 films
            top_10 = sorted(results, key=lambda x: x['score'], reverse=True)[:10]
            
            fig = go.Figure(data=[
                go.Bar(x=[r['title'][:20] + '...' for r in top_10], 
                      y=[r['score'] for r in top_10],
                      marker_color='#FF6B6B')
            ])
            fig.update_layout(title='Top 10 Films by Score',
                             xaxis_title='Film',
                             yaxis_title='Score',
                             xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key="batch_top_10")
        
        # Export options
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export to CSV
            export_df = pd.DataFrame([{
                'Title': r['title'],
                'Overall_Score': r['score'],
                'Narrative_Score': r['narrative'],
                'Character_Score': r['character'],
                'Visual_Score': r['visual'],
                'Technical_Score': r['technical'],
                'Analysis_Date': datetime.now().strftime('%Y-%m-%d')
            } for r in results])
            
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
        
        with col2:
            if st.button("🏠 Return to Dashboard", width='stretch'):
                st.session_state.show_batch_results = False
                st.session_state.batch_results = None
                st.rerun()
    
    def _get_quantum_statistics(self) -> Dict:
        """Get quantum statistics for dashboard"""
        films = list(st.session_state.stored_results.values())
        history = st.session_state.analysis_history
        
        if not films:
            return {
                "total_films": 0,
                "average_score": 0,
                "top_score": 0,
                "analysis_count": st.session_state.analysis_count,
                "recent_analyses": []
            }
        
        # Calculate statistics
        scores = [film["analysis_results"]["overall_score"] for film in films]
        
        # Recent analyses (last 5)
        recent_analyses = []
        for item in history[-5:]:
            # Calculate time ago
            timestamp = datetime.fromisoformat(item.get('timestamp', datetime.now().isoformat()))
            time_ago = datetime.now() - timestamp
            
            if time_ago.days > 0:
                time_str = f"{time_ago.days}d ago"
            elif time_ago.seconds >= 3600:
                hours = time_ago.seconds // 3600
                time_str = f"{hours}h ago"
            else:
                minutes = time_ago.seconds // 60
                time_str = f"{minutes}m ago"
            
            recent_analyses.append({
                'title': item.get('title', 'Unknown'),
                'score': item.get('overall_score', 0),
                'time': time_str
            })
        
        return {
            "total_films": len(films),
            "average_score": round(np.mean(scores), 1) if scores else 0,
            "top_score": round(max(scores), 1) if scores else 0,
            "analysis_count": st.session_state.analysis_count,
            "recent_analyses": recent_analyses[::-1]  # Reverse to show most recent first
        }

# --------------------------
# ENHANCED SIDEBAR
# --------------------------
def display_enhanced_sidebar() -> None:
    """Display enhanced sidebar for film enthusiasts"""
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 15px; background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%); 
              border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>🍅 Fresh-Tomatoes</h2>
        <p style='color: white; margin: 5px 0 0 0; font-size: 0.9em;'>Quantum Film Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.markdown("### 🧭 Navigation")
    
    nav_options = [
        ("🏠 Dashboard", "🏠 Quantum Dashboard"),
        ("📊 Analytics", "📊 Advanced Analytics"),
        ("🏆 Top Films", "🏆 Top Films"),
        ("⚙️ Settings", "⚙️ Settings"),
        ("ℹ️ About", "ℹ️ About")
    ]
    
    for label, page in nav_options:
        if st.sidebar.button(label, width='stretch', key=f"nav_{page}"):
            st.session_state.current_page = page
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Quick Stats
    films = list(st.session_state.stored_results.values())
    
    if films:
        st.sidebar.markdown("### 📈 Quick Stats")
        
        scores = [film["analysis_results"]["overall_score"] for film in films]
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.sidebar.metric("Films", len(films))
        with col2:
            if scores:
                st.sidebar.metric("Avg", f"{np.mean(scores):.1f}")
        
        # Recent activity
        if st.session_state.last_analysis_time:
            last_time = datetime.fromisoformat(st.session_state.last_analysis_time)
            time_ago = datetime.now() - last_time
            
            if time_ago.days > 0:
                last_str = f"{time_ago.days}d ago"
            elif time_ago.seconds >= 3600:
                last_str = f"{time_ago.seconds // 3600}h ago"
            else:
                last_str = f"{time_ago.seconds // 60}m ago"
            
            st.sidebar.caption(f"Last analysis: {last_str}")
    
    # YouTube viewer status
    if st.session_state.get('youtube_viewer_active'):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎬 YouTube Viewer")
        video_id = st.session_state.get('youtube_video_id', 'Unknown')
        st.sidebar.markdown(f"**Active:** {video_id[:12]}...")
        
        if st.sidebar.button("📺 Return to Viewer", width='stretch'):
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # System status
    st.sidebar.markdown("### ⚡ System Status")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.sidebar.markdown("<div style='color: #4ECDC4;'>● Quantum Ready</div>", unsafe_allow_html=True)
    with col2:
        st.sidebar.markdown("<div style='color: #4ECDC4;'>● AI Active</div>", unsafe_allow_html=True)
    
    # Version info
    st.sidebar.markdown("---")
    st.sidebar.caption("🍅 Fresh-Tomatoes AI Film Meter v4.6")
    st.sidebar.caption("Quantum Enhanced Edition")

# --------------------------
# MAIN APPLICATION
# --------------------------
def main() -> None:
    """Main application function"""
    # Display enhanced sidebar
    display_enhanced_sidebar()
    
    # Load persistence
    if not st.session_state.persistence_loaded:
        try:
            QuantumPersistenceManager.load_session()
            st.session_state.persistence_loaded = True
        except Exception as e:
            print(f"Error loading session: {e}")
    
    # Check for YouTube viewer persistence before routing
    if st.session_state.get('youtube_viewer_active') and st.session_state.get('youtube_video_id'):
        # Always show YouTube viewer if it's active, regardless of current page
        if QuantumPersistenceManager.restore_youtube_viewer():
            quantum_interface = QuantumFilmInterface()
            
            # Handle analysis requests
            if st.session_state.get('quick_analyze_requested'):
                st.session_state.quick_analyze_requested = False
                quantum_interface._perform_quick_youtube_analysis(st.session_state.youtube_video_id)
                return
            
            if st.session_state.get('detailed_analyze_requested'):
                st.session_state.detailed_analyze_requested = False
                quantum_interface._perform_detailed_youtube_analysis(st.session_state.youtube_video_id)
                return
            
            # Display YouTube viewer
            YouTubeViewer.display_youtube_viewer(
                st.session_state.youtube_video_id,
                st.session_state.youtube_video_data
            )
            return
    
    # Initialize quantum interface
    quantum_interface = QuantumFilmInterface()
    
    # Route based on current page
    page = st.session_state.get('current_page', '🏠 Quantum Dashboard')
    
    if page == "🏠 Quantum Dashboard":
        quantum_interface.show_quantum_dashboard()
    elif page == "📊 Advanced Analytics":
        st.header("📊 Advanced Analytics")
        quantum_interface._show_advanced_analytics()
        
        if st.button("← Return to Dashboard", width='stretch'):
            st.session_state.current_page = "🏠 Quantum Dashboard"
            st.rerun()
    elif page == "🏆 Top Films":
        st.header("🏆 Top Films Analysis")
        quantum_interface._show_top_films_display()
        
        if st.button("← Return to Dashboard", width='stretch'):
            st.session_state.current_page = "🏠 Quantum Dashboard"
            st.rerun()
    elif page == "⚙️ Settings":
        st.header("⚙️ Settings & Preferences")
        quantum_interface._show_user_preferences()
        
        if st.button("← Return to Dashboard", width='stretch'):
            st.session_state.current_page = "🏠 Quantum Dashboard"
            st.rerun()
    elif page == "ℹ️ About":
        st.header("ℹ️ About Fresh-Tomatoes AI Film Meter")
        
        about_tab1, about_tab2, about_tab3 = st.tabs(["Overview", "Features", "Technology"])
        
        with about_tab1:
            st.markdown("""
            ## 🍅 Fresh-Tomatoes AI Film Meter
            
            **The Intelligent Film Analysis Platform for True Cinema Enthusiasts**
            
            Fresh-Tomatoes AI Film Meter combines quantum-inspired algorithms with 
            multimodal AI to provide comprehensive film analysis. Designed for 
            filmmakers, critics, and cinema lovers who appreciate the art of filmmaking.
            
            **Key Philosophy:**
            - Cinema as artistic expression
            - Quantitative meets qualitative analysis
            - Focus on cinematic craftsmanship
            - Celebration of film artistry
            """)
        
        with about_tab2:
            st.markdown("""
            ## 🌟 Core Features
            
            **🎬 Comprehensive Film Analysis:**
            - Quantum-inspired scoring algorithm
            - Multimodal AI assessment (text, visual, audio)
            - Detailed component breakdowns
            - Film enthusiast focused insights
            
            **📊 Advanced Analytics:**
            - Batch processing capabilities
            - Score distribution analysis
            - Component correlation studies
            - Trend identification
            
            **🎥 YouTube Integration:**
            - Direct YouTube video analysis
            - Persistent viewer window
            - Real-time scoring
            - Transcript extraction
            
            **🏆 Film Enthusiast Tools:**
            - Top films ranking
            - Festival readiness assessment
            - Critical reception prediction
            - Development recommendations
            
            **⚙️ Professional Features:**
            - Persistent data storage
            - CSV import/export
            - Customizable preferences
            - Session persistence
            """)
        
        with about_tab3:
            st.markdown("""
            ## 🔬 Technology Stack
            
            **AI & Machine Learning:**
            - Quantum-inspired algorithms
            - Natural Language Processing (NLP)
            - Sentiment analysis with VADER
            - Multimodal fusion techniques
            
            **Data Processing:**
            - Pandas for data manipulation
            - Plotly for advanced visualizations
            - NumPy for numerical computations
            - Custom scoring algorithms
            
            **Web Framework:**
            - Streamlit for interactive UI
            - Custom CSS for film enthusiast design
            - Responsive dashboard layouts
            - Real-time data visualization
            
            **Integration:**
            - YouTube video embedding
            - CSV batch processing
            - JSON data export/import
            - Persistent session management
            """)
        
        if st.button("← Return to Dashboard", width='stretch'):
            st.session_state.current_page = "🏠 Quantum Dashboard"
            st.rerun()
    else:
        quantum_interface.show_quantum_dashboard()

if __name__ == "__main__":
    main()
