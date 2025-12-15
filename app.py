"""
FlickFinder AI - Advanced Film Analysis Platform
Version: 4.0 Cinematic Intelligence Ecosystem
Description: State-of-the-art film analysis with YouTube integration, cultural context analysis,
             comprehensive scoring, radial graphing, ePortfolio system, and cinematic visualizations.
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
from plotly.subplots import make_subplots
from collections import Counter
import textwrap
import time
import io
import base64
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import math
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# CONFIGURATION & SETUP
# --------------------------
st.set_page_config(
    page_title="FlickFinder AI 🎬 - Cinematic Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/flickfinder-ai',
        'Report a bug': 'https://github.com/flickfinder-ai/issues',
        'About': "### FlickFinder AI v4.0\nState-of-the-art film analysis using AI and machine learning with cinematic intelligence."
    }
)

# Add custom CSS for enhanced UI
st.markdown("""
<style>
    /* Cinematic gradient backgrounds */
    .cinematic-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .holographic-card {
        background: linear-gradient(135deg, rgba(20, 30, 48, 0.9) 0%, rgba(36, 59, 85, 0.9) 100%);
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .holographic-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: 0.5s;
    }
    
    .holographic-card:hover::before {
        left: 100%;
    }
    
    .tech-panel {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border: 1px solid rgba(0, 200, 255, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .radial-bg {
        background: radial-gradient(circle at 30% 30%, rgba(102, 126, 234, 0.1) 0%, 
                                   rgba(118, 75, 162, 0.05) 25%, transparent 70%);
    }
    
    .cinematic-score {
        font-family: 'Arial Black', sans-serif;
        font-size: 2.5em;
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF8C00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: calc(200px + 100%) 0; }
    }
    
    .grid-cell {
        background: rgba(0, 30, 60, 0.7);
        padding: 15px;
        border-radius: 8px;
        border: 1px solid rgba(0, 150, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .grid-cell:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 150, 255, 0.3);
        border-color: rgba(0, 200, 255, 0.5);
    }
    
    .film-card {
        background: linear-gradient(135deg, rgba(30, 40, 60, 0.9) 0%, rgba(20, 30, 50, 0.9) 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(0, 150, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .film-card:hover {
        border-color: rgba(0, 200, 255, 0.5);
        box-shadow: 0 5px 15px rgba(0, 200, 255, 0.2);
    }
    
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8em;
        margin: 2px;
    }
    
    .elite-badge { background: linear-gradient(45deg, #FFD700, #FFA500); color: #000; }
    .excellent-badge { background: linear-gradient(45deg, #00FFAA, #00CC88); color: #000; }
    .strong-badge { background: linear-gradient(45deg, #AAFF00, #88CC00); color: #000; }
    .good-badge { background: linear-gradient(45deg, #FFFF00, #FFCC00); color: #000; }
    .developing-badge { background: linear-gradient(45deg, #FFAA00, #FF8800); color: #000; }
    
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 150, 255, 0.3);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #764ba2, #f093fb);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# SESSION STATE INITIALIZATION
# --------------------------
class SessionManager:
    """Manages session state initialization and persistence"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables"""
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
            'show_breakdown': True,
            'current_tab': 'youtube',
            'persistence_loaded': False,
            'eportfolio_view': 'cinematic',
            'show_tech_view': False,
            'selected_film_ids': [],
            'comparison_mode': False,
            'holographic_mode': True,
            'initialized': True,
        }
        
        for key, default in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

# --------------------------
# UTILITY FUNCTIONS
# --------------------------
class TextProcessor:
    """Handles text processing utilities"""
    
    @staticmethod
    @st.cache_resource
    def initialize_nltk():
        """Initialize NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        return SentimentIntensityAnalyzer()
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',
            r'(?:youtu\.be\/)([^&\n?#]+)',
            r'(?:youtube\.com\/embed\/)([^&\n?#]+)',
            r'(?:youtube\.com\/v\/)([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    @staticmethod
    def get_youtube_transcript(video_id: str) -> Optional[str]:
        """Get transcript from YouTube video"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([t['text'] for t in transcript_list])
            return transcript
        except Exception as e:
            st.warning(f"Could not fetch transcript: {str(e)}")
            return None

# --------------------------
# CINEMATIC VISUALIZATION ENGINE
# --------------------------
class CinematicVisualizationEngine:
    """Advanced visualization engine with radial graphing and holographic displays"""
    
    @staticmethod
    def create_cinematic_radar_chart(scores: Dict[str, float], title: str = "Cinematic Analysis") -> go.Figure:
        """Create an advanced radar chart with cinematic styling"""
        if not scores:
            return None
            
        categories = list(scores.keys())
        values = list(scores.values())
        
        # Close the loop
        categories_display = [cat.replace('_', ' ').title() for cat in categories]
        categories_display.append(categories_display[0])
        values.append(values[0])
        
        fig = go.Figure()
        
        # Main radar trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_display,
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='rgb(102, 126, 234)', width=3),
            name='Score',
            hoverinfo='text+value',
            text=[f"{cat}: {val}/5.0" for cat, val in zip(categories_display[:-1], values[:-1])],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add reference lines
        for i in range(1, 6):
            fig.add_trace(go.Scatterpolar(
                r=[i] * (len(categories) + 1),
                theta=categories_display,
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.1)', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=['1', '2', '3', '4', '5'],
                    tickfont=dict(size=12, color='#FFFFFF'),
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    linecolor='rgba(255, 255, 255, 0.3)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    linecolor='rgba(255, 255, 255, 0.3)',
                    rotation=90,
                    direction='clockwise'
                ),
                bgcolor='rgba(10, 15, 30, 0.7)'
            ),
            title=dict(
                text=f'<b>{title}</b>',
                font=dict(size=20, color='#FFFFFF', family="Arial Black"),
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            height=400,
            margin=dict(l=50, r=50, t=60, b=50),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#FFFFFF')
        )
        
        return fig
    
    @staticmethod
    def create_philosophical_radial_chart(insights: Dict[str, float]) -> Optional[go.Figure]:
        """Create a radial chart for philosophical insights"""
        if not insights:
            return None
            
        themes = list(insights.keys())
        scores = list(insights.values())
        
        fig = go.Figure()
        
        # Create radial bars
        fig.add_trace(go.Barpolar(
            r=scores,
            theta=themes,
            width=[0.8] * len(themes),
            marker=dict(
                color=scores,
                colorscale='Viridis',
                line=dict(color='#FFFFFF', width=1)
            ),
            hoverinfo='text',
            text=[f"{theme}: {score:.1f}" for theme, score in zip(themes, scores)],
            hovertemplate='<b>%{theta}</b><br>Depth: %{r:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(scores) * 1.2 if scores else 1],
                    showticklabels=True,
                    tickfont=dict(size=10, color='#FFFFFF')
                ),
                angularaxis=dict(
                    direction='clockwise',
                    rotation=90,
                    tickfont=dict(size=11, color='#FFFFFF')
                ),
                bgcolor='rgba(15, 25, 40, 0.8)'
            ),
            title=dict(
                text='<b>Philosophical Depth Analysis</b>',
                font=dict(size=16, color='#FFFFFF', family="Arial"),
                x=0.5
            ),
            height=350,
            margin=dict(l=50, r=50, t=60, b=50),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )
        
        return fig
    
    @staticmethod
    def create_tech_breakdown_chart(weights: Dict[str, float], scores: Dict[str, float]) -> Optional[go.Figure]:
        """Create technology breakdown visualization"""
        if not weights or not scores:
            return None
            
        components = list(weights.keys())
        weight_values = [w * 100 for w in weights.values()]  # Convert to percentages
        score_values = [scores.get(c, 0) for c in components]
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'pie'}, {'type': 'bar'}]],
            subplot_titles=('<b>Algorithm Weight Distribution</b>', '<b>Component Performance</b>'),
            horizontal_spacing=0.15
        )
        
        # Pie chart for weights
        fig.add_trace(
            go.Pie(
                labels=[c.replace('_', ' ').title() for c in components],
                values=weight_values,
                hole=0.4,
                marker=dict(colors=px.colors.sequential.Viridis),
                textinfo='label+percent',
                textposition='outside',
                hoverinfo='label+value+percent',
                name="Weights"
            ),
            row=1, col=1
        )
        
        # Bar chart for scores
        fig.add_trace(
            go.Bar(
                x=[c.replace('_', ' ').title() for c in components],
                y=score_values,
                marker=dict(
                    color=score_values,
                    colorscale='Viridis',
                    line=dict(color='#FFFFFF', width=1)
                ),
                text=[f"{s:.1f}" for s in score_values],
                textposition='auto',
                hoverinfo='x+y'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#FFFFFF'),
            title=dict(
                text='<b>AI Scoring Engine Breakdown</b>',
                font=dict(size=20, color='#FFFFFF', family="Arial Black"),
                x=0.5
            )
        )
        
        fig.update_xaxes(title_text="Components", row=1, col=2)
        fig.update_yaxes(title_text="Score /5.0", row=1, col=2, range=[0, 5.5])
        
        return fig
    
    @staticmethod
    def create_comparison_radar(films_data: List[Dict]) -> Optional[go.Figure]:
        """Create comparison radar chart for multiple films"""
        if len(films_data) < 2:
            return None
            
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD93D', '#FF8E53']
        
        for idx, film in enumerate(films_data[:6]):  # Limit to 6 films for clarity
            scores = film.get('cinematic_scores', {})
            if not scores:
                continue
                
            categories = list(scores.keys())
            values = list(scores.values())
            categories_display = [cat.replace('_', ' ').title() for cat in categories]
            categories_display.append(categories_display[0])
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_display,
                fill='toself' if len(films_data) <= 3 else 'none',
                fillcolor=f'rgba{tuple(int(colors[idx].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                line=dict(color=colors[idx], width=2),
                name=film.get('title', f'Film {idx+1}')[:20],
                opacity=0.8 if len(films_data) <= 3 else 1.0
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5],
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=['1', '2', '3', '4', '5'],
                    tickfont=dict(size=10, color='#FFFFFF')
                ),
                angularaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    linecolor='rgba(255, 255, 255, 0.3)'
                ),
                bgcolor='rgba(10, 15, 30, 0.7)'
            ),
            title=dict(
                text='<b>Multi-Film Comparison Analysis</b>',
                font=dict(size=18, color='#FFFFFF'),
                x=0.5
            ),
            height=500,
            margin=dict(l=50, r=50, t=60, b=50),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#FFFFFF'),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(0, 0, 0, 0.5)',
                bordercolor='rgba(255, 255, 255, 0.2)'
            )
        )
        
        return fig
    
    @staticmethod
    def create_score_gauge(score: float, title: str = "Cinematic Score") -> go.Figure:
        """Create a gauge chart for scores"""
        # Determine color based on score
        if score >= 4.5:
            color = "#00FFFF"
        elif score >= 4.0:
            color = "#00FFAA"
        elif score >= 3.5:
            color = "#AAFF00"
        elif score >= 3.0:
            color = "#FFFF00"
        else:
            color = "#FFAA00"
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20, 'color': 'white'}},
            number={'font': {'size': 40, 'color': color, 'family': "Arial Black"}},
            gauge={
                'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': 'white'},
                'bar': {'color': color},
                'bgcolor': 'rgba(0, 0, 0, 0)',
                'borderwidth': 2,
                'bordercolor': color,
                'steps': [
                    {'range': [0, 2.5], 'color': 'rgba(255, 0, 0, 0.1)'},
                    {'range': [2.5, 4], 'color': 'rgba(255, 255, 0, 0.1)'},
                    {'range': [4, 5], 'color': 'rgba(0, 255, 0, 0.1)'}
                ]
            }
        ))
        
        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font={'color': 'white'}
        )
        
        return fig

# --------------------------
# PERSISTENCE MANAGER
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
        st.session_state.analysis_count = len(st.session_state.analysis_history)
        
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
        st.session_state.selected_film_ids = []
    
    @staticmethod
    def get_all_films() -> List[Dict]:
        """Get all films"""
        return list(st.session_state.stored_results.values())
    
    @staticmethod
    def get_analytics_data() -> Optional[pd.DataFrame]:
        """Get comprehensive analytics data"""
        history = st.session_state.analysis_history
        if not history:
            return None
        
        df = pd.DataFrame(history)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['time_of_day'] = df['timestamp'].dt.hour
        
        return df

# --------------------------
# GENRE DETECTOR
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
# CULTURAL CONTEXT ANALYZER
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
# FILM SCORER
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
        
        # Get detected genre
        genre_context = analysis_results.get('genre_context', {})
        detected_genre = genre_context.get('primary_genre', '')
        if isinstance(detected_genre, dict):
            detected_genre = detected_genre.get('primary_genre', '')
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
# FILM ANALYSIS ENGINE
# --------------------------
class FilmAnalysisEngine:
    """Main engine for comprehensive film analysis"""
    
    def __init__(self):
        self.genre_detector = SmartGenreDetector()
        self.cultural_analyzer = CulturalContextAnalyzer()
        self.film_scorer = FilmSpecificScorer()
        self.sentiment_analyzer = TextProcessor.initialize_nltk()
        self.persistence = PersistenceManager()
        self.viz_engine = CinematicVisualizationEngine()
    
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
        
        return {
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
            'narrative_arc_analysis': analysis_results.get('narrative_structure', {}),
            'character_ecosystem': analysis_results.get('character_analysis', {})
        }
    
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
            'ai_tool_suggestions': []
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
            'ai_tool_suggestions': [{'tool': 'Error Recovery', 'purpose': 'Issue diagnosis', 'benefit': 'Improved analysis stability'}]
        }

# --------------------------
# ePORTFOLIO SYSTEM
# --------------------------
class EnhancedEPortfolioSystem:
    """Enhanced ePortfolio system with cinematic visualization"""
    
    def __init__(self, persistence, viz_engine):
        self.persistence = persistence
        self.viz_engine = viz_engine
    
    def show_portfolio(self) -> None:
        """Display the enhanced ePortfolio dashboard"""
        st.header("🎬 Cinematic ePortfolio")
        st.markdown("---")
        
        # Get all films
        all_films = self.persistence.get_all_films()
        
        if not all_films:
            st.info("No films in your portfolio yet. Analyze some films to build your collection!")
            if st.button("← Back to Dashboard"):
                st.session_state.current_page = "🏠 Dashboard"
                st.rerun()
            return
        
        # Portfolio stats
        stats = self._calculate_portfolio_stats(all_films)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self._create_portfolio_stat("Total Films", stats['total'], "#667eea")
        with col2:
            self._create_portfolio_stat("Avg Score", f"{stats['avg_score']:.1f}", "#764ba2")
        with col3:
            self._create_portfolio_stat("Top Genre", stats['top_genre'], "#f093fb")
        with col4:
            self._create_portfolio_stat("Elite Films", stats['elite_count'], "#FFD700")
        
        # View selector
        view_options = ['Cinematic Grid', 'Score Timeline', 'Genre Analysis', 'Cultural Focus']
        selected_view = st.selectbox("Portfolio View:", view_options)
        
        if selected_view == 'Cinematic Grid':
            self._show_cinematic_grid(all_films)
        elif selected_view == 'Score Timeline':
            self._show_score_timeline(all_films)
        elif selected_view == 'Genre Analysis':
            self._show_genre_analysis(all_films)
        elif selected_view == 'Cultural Focus':
            self._show_cultural_analysis(all_films)
        
        # Back button
        if st.button("← Back to Dashboard", use_container_width=True):
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()
    
    def _calculate_portfolio_stats(self, films: List) -> Dict:
        """Calculate portfolio statistics"""
        scores = [f['analysis_results']['overall_score'] for f in films]
        
        # Genre distribution
        genre_counter = Counter()
        for film in films:
            genre = film['analysis_results'].get('genre_insights', {}).get('primary_genre', 'Unknown')
            if isinstance(genre, dict):
                genre = genre.get('primary_genre', 'Unknown')
            genre_counter[genre] += 1
        
        # Elite films (score >= 4.0)
        elite_films = [f for f in films if f['analysis_results']['overall_score'] >= 4.0]
        
        return {
            'total': len(films),
            'avg_score': np.mean(scores) if scores else 0,
            'top_genre': genre_counter.most_common(1)[0][0] if genre_counter else 'N/A',
            'elite_count': len(elite_films)
        }
    
    def _create_portfolio_stat(self, label: str, value: Any, color: str) -> None:
        """Create portfolio statistic card"""
        html = f"""
        <div class="grid-cell" style="text-align: center;">
            <div style="color: #AAAAAA; font-size: 0.9em; margin-bottom: 5px;">{label}</div>
            <div style="color: {color}; font-size: 1.8em; font-weight: bold; margin: 10px 0;">
                {value}
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    def _show_cinematic_grid(self, films: List) -> None:
        """Display films in cinematic grid"""
        # Sort by score
        sorted_films = sorted(films, key=lambda x: x['analysis_results']['overall_score'], reverse=True)
        
        # Create grid
        cols = st.columns(4)
        for idx, film in enumerate(sorted_films[:12]):  # Limit to 12 films
            with cols[idx % 4]:
                self._display_film_portfolio_card(film, idx)
    
    def _display_film_portfolio_card(self, film: Dict, idx: int) -> None:
        """Display film portfolio card"""
        film_data = film['film_data']
        analysis = film['analysis_results']
        
        title = film_data.get('title', 'Unknown')[:20]
        score = analysis['overall_score']
        genre = analysis.get('genre_insights', {}).get('primary_genre', 'Unknown')
        if isinstance(genre, dict):
            genre = genre.get('primary_genre', 'Unknown')
        
        # Color based on score
        if score >= 4.5:
            color = "#00FFFF"
            status = "ELITE"
        elif score >= 4.0:
            color = "#00FFAA"
            status = "EXCELLENT"
        elif score >= 3.5:
            color = "#AAFF00"
            status = "STRONG"
        elif score >= 3.0:
            color = "#FFFF00"
            status = "GOOD"
        else:
            color = "#FFAA00"
            status = "DEVELOPING"
        
        # Create card
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{title}**")
        with col2:
            st.markdown(f"<span style='color: {color}; font-weight: bold;'>{score:.1f}</span>", 
                       unsafe_allow_html=True)
        
        st.markdown(f"<small style='color: #666;'>{genre}</small>", unsafe_allow_html=True)
        
        if st.button("View Details", key=f"view_{idx}", use_container_width=True):
            st.session_state.current_results_display = analysis
            st.session_state.show_results_page = True
            st.session_state.current_analysis_id = film.get('film_id', idx)
            st.session_state.current_page = "📊 Results"
            st.rerun()
        
        st.divider()
    
    def _show_score_timeline(self, films: List) -> None:
        """Display score timeline visualization"""
        # Prepare timeline data
        timeline_data = []
        for film in films:
            film_data = film['film_data']
            analysis = film['analysis_results']
            
            timestamp = film_data.get('timestamp', '')
            if not timestamp:
                timestamp = (datetime.now() - timedelta(days=len(timeline_data))).isoformat()
            
            try:
                date = pd.Timestamp(timestamp).date()
                timeline_data.append({
                    'date': date,
                    'score': analysis['overall_score'],
                    'title': film_data.get('title', 'Unknown'),
                    'genre': analysis.get('genre_insights', {}).get('primary_genre', 'Unknown')
                })
            except:
                continue
        
        if timeline_data:
            df = pd.DataFrame(timeline_data)
            df = df.sort_values('date')
            
            # Create timeline chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['score'],
                mode='lines+markers',
                line=dict(color='#00FFFF', width=3),
                marker=dict(size=10, color=df['score'], colorscale='Viridis'),
                text=df['title'],
                hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Score: %{y:.1f}<extra></extra>'
            ))
            
            # Add trend line
            z = np.polyfit(range(len(df)), df['score'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=p(range(len(df))),
                mode='lines',
                line=dict(color='#FFD700', width=2, dash='dash'),
                name='Trend'
            ))
            
            fig.update_layout(
                title=dict(
                    text='<b>Score Timeline Evolution</b>',
                    font=dict(size=18, color='white')
                ),
                height=400,
                paper_bgcolor='rgba(0, 0, 0, 0)',
                plot_bgcolor='rgba(20, 30, 48, 0.7)',
                font=dict(color='white'),
                xaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    title='Date'
                ),
                yaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    title='Score',
                    range=[0, 5.5]
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Highest Score", f"{df['score'].max():.1f}")
            with col2:
                st.metric("Lowest Score", f"{df['score'].min():.1f}")
            with col3:
                st.metric("Average Score", f"{df['score'].mean():.1f}")
    
    def _show_genre_analysis(self, films: List) -> None:
        """Display genre analysis visualization"""
        genre_counter = Counter()
        genre_scores = {}
        
        for film in films:
            analysis = film['analysis_results']
            genre = analysis.get('genre_insights', {}).get('primary_genre', 'Unknown')
            if isinstance(genre, dict):
                genre = genre.get('primary_genre', 'Unknown')
            
            genre_counter[genre] += 1
            
            if genre not in genre_scores:
                genre_scores[genre] = []
            genre_scores[genre].append(analysis['overall_score'])
        
        if not genre_counter:
            st.info("No genre data available")
            return
        
        # Calculate average scores per genre
        avg_scores = {genre: np.mean(scores) for genre, scores in genre_scores.items()}
        
        # Create visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre distribution pie chart
            fig1 = go.Figure(data=[
                go.Pie(
                    labels=list(genre_counter.keys()),
                    values=list(genre_counter.values()),
                    hole=0.4,
                    marker=dict(colors=px.colors.qualitative.Set3),
                    textinfo='label+percent',
                    hoverinfo='label+value+percent'
                )
            ])
            
            fig1.update_layout(
                title=dict(
                    text='<b>Genre Distribution</b>',
                    font=dict(size=16, color='white')
                ),
                height=350,
                paper_bgcolor='rgba(0, 0, 0, 0)',
                plot_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Genre scores bar chart
            genres_sorted = sorted(avg_scores.keys(), key=lambda g: avg_scores[g], reverse=True)
            fig2 = go.Figure(data=[
                go.Bar(
                    x=genres_sorted,
                    y=[avg_scores[g] for g in genres_sorted],
                    marker=dict(
                        color=[avg_scores[g] for g in genres_sorted],
                        colorscale='Viridis'
                    ),
                    text=[f"{avg_scores[g]:.1f}" for g in genres_sorted],
                    textposition='auto'
                )
            ])
            
            fig2.update_layout(
                title=dict(
                    text='<b>Average Scores by Genre</b>',
                    font=dict(size=16, color='white')
                ),
                height=350,
                paper_bgcolor='rgba(0, 0, 0, 0)',
                plot_bgcolor='rgba(20, 30, 48, 0.7)',
                font=dict(color='white'),
                xaxis=dict(tickangle=45),
                yaxis=dict(range=[0, 5.5], title='Average Score')
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    def _show_cultural_analysis(self, films: List) -> None:
        """Display cultural analysis visualization"""
        cultural_films = []
        cultural_scores = []
        cultural_themes = Counter()
        
        for film in films:
            analysis = film['analysis_results']
            cultural_insights = analysis.get('cultural_insights', {})
            
            if cultural_insights.get('is_culturally_relevant'):
                cultural_films.append(film)
                cultural_scores.append(analysis['overall_score'])
                
                # Count themes
                themes = cultural_insights.get('primary_themes', [])
                for theme in themes:
                    cultural_themes[theme] += 1
        
        if not cultural_films:
            st.info("No culturally relevant films found")
            return
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Culturally Relevant Films", len(cultural_films))
        with col2:
            st.metric("Average Score", f"{np.mean(cultural_scores):.1f}")
        with col3:
            top_theme = cultural_themes.most_common(1)[0][0] if cultural_themes else "N/A"
            st.metric("Top Theme", top_theme)
        
        # Display culturally relevant films
        st.subheader("Culturally Relevant Films")
        cols = st.columns(3)
        for idx, film in enumerate(cultural_films[:6]):
            with cols[idx % 3]:
                film_data = film['film_data']
                analysis = film['analysis_results']
                
                st.markdown(f"""
                <div class="grid-cell">
                    <h4 style="color: white; margin-bottom: 5px;">
                        {film_data.get('title', 'Unknown')[:20]}
                    </h4>
                    <div style="color: #00FFFF; font-size: 1.2em; font-weight: bold;">
                        {analysis['overall_score']:.1f}
                    </div>
                    <div style="color: #AAAAAA; font-size: 0.9em; margin-top: 5px;">
                        {', '.join(analysis.get('cultural_insights', {}).get('primary_themes', []))[:30]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Themes visualization
        if cultural_themes:
            st.subheader("Cultural Themes Distribution")
            themes_df = pd.DataFrame({
                'Theme': list(cultural_themes.keys()),
                'Count': list(cultural_themes.values())
            }).sort_values('Count', ascending=False)
            
            fig = px.bar(
                themes_df,
                x='Theme',
                y='Count',
                color='Count',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0, 0, 0, 0)',
                plot_bgcolor='rgba(20, 30, 48, 0.7)',
                font=dict(color='white'),
                xaxis=dict(tickangle=45)
            )
            
            st.plotly_chart(fig, use_container_width=True)

# --------------------------
# COMPARISON SYSTEM
# --------------------------
class EnhancedComparisonSystem:
    """System for comparing multiple films"""
    
    def __init__(self, persistence, viz_engine):
        self.persistence = persistence
        self.viz_engine = viz_engine
    
    def show_comparison(self) -> None:
        """Display film comparison interface"""
        st.title("🎬 Film Comparison Analysis")
        st.markdown("---")
        
        # Get all films
        all_films = self.persistence.get_all_films()
        
        if not all_films:
            st.info("No films to compare. Analyze some films first!")
            if st.button("← Back to Dashboard"):
                st.session_state.current_page = "🏠 Dashboard"
                st.rerun()
            return
        
        # Film selection
        st.subheader("Select Films to Compare")
        
        film_options = {}
        for film in all_films:
            film_data = film['film_data']
            title = film_data.get('title', 'Unknown')
            film_id = film.get('film_id', hash(title))
            film_options[f"{title} (Score: {film['analysis_results']['overall_score']:.1f})"] = film_id
        
        selected_films = st.multiselect(
            "Choose 2-6 films to compare:",
            options=list(film_options.keys()),
            default=list(film_options.keys())[:min(3, len(film_options))]
        )
        
        if len(selected_films) < 2:
            st.warning("Please select at least 2 films for comparison.")
            return
        
        # Get selected film data
        selected_data = []
        for film_label in selected_films:
            film_id = film_options[film_label]
            for film in all_films:
                if film.get('film_id') == film_id or hash(film['film_data'].get('title', '')) == film_id:
                    selected_data.append(film)
                    break
        
        # Display comparison
        if selected_data:
            self._display_comparison(selected_data)
        
        # Back button
        if st.button("← Back to Dashboard", use_container_width=True):
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()
    
    def _display_comparison(self, films: List[Dict]) -> None:
        """Display film comparison"""
        # Overall scores comparison
        st.subheader("Overall Scores Comparison")
        
        scores_data = []
        for film in films:
            film_data = film['film_data']
            analysis = film['analysis_results']
            
            scores_data.append({
                'Film': film_data.get('title', 'Unknown')[:20],
                'Score': analysis['overall_score'],
                'Genre': analysis.get('genre_insights', {}).get('primary_genre', 'Unknown'),
                'Cultural Score': analysis.get('cultural_insights', {}).get('relevance_score', 0) * 5
            })
        
        df_scores = pd.DataFrame(scores_data)
        
        # Bar chart comparison
        fig = px.bar(
            df_scores,
            x='Film',
            y='Score',
            color='Score',
            color_continuous_scale='Viridis',
            text='Score',
            title='Overall Score Comparison'
        )
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(20, 30, 48, 0.7)',
            font=dict(color='white'),
            xaxis=dict(tickangle=45)
        )
        
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart comparison
        st.subheader("Cinematic Elements Comparison")
        
        radar_data = []
        for film in films:
            film_data = film['film_data']
            analysis = film['analysis_results']
            
            radar_entry = {
                'title': film_data.get('title', 'Unknown')[:15],
                'scores': analysis.get('cinematic_scores', {})
            }
            radar_data.append(radar_entry)
        
        fig = self.viz_engine.create_comparison_radar(radar_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("Detailed Comparison")
        
        comparison_cols = st.columns(len(films))
        for idx, film in enumerate(films):
            with comparison_cols[idx]:
                film_data = film['film_data']
                analysis = film['analysis_results']
                
                st.markdown(f"""
                <div class="holographic-card">
                    <h4 style="color: #00FFFF; text-align: center;">
                        {film_data.get('title', 'Unknown')[:20]}
                    </h4>
                    <div style="text-align: center;">
                        <span style="color: #FFD700; font-size: 1.5em; font-weight: bold;">
                            {analysis['overall_score']:.1f}
                        </span>
                    </div>
                    <hr style="border-color: rgba(255,255,255,0.2);">
                """, unsafe_allow_html=True)
                
                # Genre
                genre = analysis.get('genre_insights', {}).get('primary_genre', 'Unknown')
                st.markdown(f"**Genre:** {genre}")
                
                # Cultural relevance
                cultural = analysis.get('cultural_insights', {})
                if cultural.get('is_culturally_relevant'):
                    st.markdown("**Cultural:** ✅ Relevant")
                else:
                    st.markdown("**Cultural:** ⚪ Standard")
                
                # Strengths
                strengths = analysis.get('strengths', [])[:2]
                if strengths:
                    st.markdown("**Strengths:**")
                    for strength in strengths:
                        st.markdown(f"• {strength}")
                
                st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# RESULTS DISPLAY
# --------------------------
class EnhancedResultsDisplay:
    """Display enhanced analysis results with cinematic visuals"""
    
    def __init__(self, viz_engine):
        self.viz_engine = viz_engine
    
    def show_results(self, results: Dict) -> None:
        """Display enhanced analysis results"""
        if not results:
            st.warning("No results to display")
            return
        
        # Main results header
        st.title(f"🎬 {results.get('film_title', 'Film Analysis')}")
        st.markdown("---")
        
        # Score and summary section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self._show_score_card(results)
        
        with col2:
            self._show_summary_card(results)
        
        # Visualizations section
        st.markdown("## 📊 Cinematic Analysis Visualizations")
        
        viz_tabs = st.tabs(["Radar Analysis", "Component Breakdown", "Philosophical Insights"])
        
        with viz_tabs[0]:
            self._show_radar_analysis(results)
        
        with viz_tabs[1]:
            self._show_component_breakdown(results)
        
        with viz_tabs[2]:
            self._show_philosophical_insights(results)
        
        # Detailed analysis section
        st.markdown("## 🔍 Detailed Analysis")
        
        detail_tabs = st.tabs(["Strengths & Weaknesses", "Recommendations", "Audience & Festival", "AI Insights"])
        
        with detail_tabs[0]:
            self._show_strengths_weaknesses(results)
        
        with detail_tabs[1]:
            self._show_recommendations(results)
        
        with detail_tabs[2]:
            self._show_audience_festival(results)
        
        with detail_tabs[3]:
            self._show_ai_insights(results)
        
        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("← Back to Dashboard", use_container_width=True):
                st.session_state.current_page = "🏠 Dashboard"
                st.session_state.show_results_page = False
                st.rerun()
        
        with col2:
            if st.button("📂 Save to Portfolio", use_container_width=True):
                st.success("Saved to portfolio!")
        
        with col3:
            if st.button("🔄 Compare with Others", use_container_width=True):
                st.session_state.current_page = "⚖️ Compare Films"
                st.rerun()
    
    def _show_score_card(self, results: Dict) -> None:
        """Display score card with cinematic styling"""
        score = results['overall_score']
        
        # Create gauge chart
        fig = self.viz_engine.create_score_gauge(score, "Cinematic Score")
        st.plotly_chart(fig, use_container_width=True)
        
        # Determine rating
        if score >= 4.5:
            rating = "ELITE"
            icon = "🔥"
            color = "#00FFFF"
        elif score >= 4.0:
            rating = "EXCELLENT"
            icon = "⭐"
            color = "#00FFAA"
        elif score >= 3.5:
            rating = "STRONG"
            icon = "💫"
            color = "#AAFF00"
        elif score >= 3.0:
            rating = "GOOD"
            icon = "✨"
            color = "#FFFF00"
        else:
            rating = "DEVELOPING"
            icon = "🌱"
            color = "#FFAA00"
        
        # Rating badge
        st.markdown(f"""
        <div style="text-align: center; margin-top: -20px;">
            <span style="background: {color}20; color: {color}; padding: 8px 20px; 
                      border-radius: 20px; border: 1px solid {color}80; font-weight: bold;">
                {icon} {rating}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    def _show_summary_card(self, results: Dict) -> None:
        """Display summary card"""
        st.markdown("""
        <div class="holographic-card">
            <h3 style="color: #00FFFF; margin-top: 0;">📋 Smart Summary</h3>
        """, unsafe_allow_html=True)
        
        st.markdown(results.get('smart_summary', 'No summary available.'))
        
        # Genre and cultural info
        genre_info = results.get('genre_insights', {})
        cultural_info = results.get('cultural_insights', {})
        
        col1, col2 = st.columns(2)
        with col1:
            if isinstance(genre_info, dict):
                primary_genre = genre_info.get('primary_genre', 'Unknown')
                if isinstance(primary_genre, dict):
                    primary_genre = primary_genre.get('primary_genre', 'Unknown')
                st.metric("Primary Genre", primary_genre)
        
        with col2:
            if cultural_info.get('is_culturally_relevant'):
                st.metric("Cultural Relevance", "High", delta="Relevant")
            else:
                st.metric("Cultural Relevance", "Standard")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _show_radar_analysis(self, results: Dict) -> None:
        """Display radar analysis"""
        cinematic_scores = results.get('cinematic_scores', {})
        
        if cinematic_scores:
            fig = self.viz_engine.create_cinematic_radar_chart(
                cinematic_scores, 
                "Cinematic Element Analysis"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cinematic scores available for radar analysis.")
    
    def _show_component_breakdown(self, results: Dict) -> None:
        """Display component breakdown"""
        scoring_breakdown = results.get('scoring_breakdown', {})
        
        if scoring_breakdown:
            weights = scoring_breakdown.get('applied_weights', {})
            weighted_scores = scoring_breakdown.get('weighted_scores', {})
            
            if weights and weighted_scores:
                fig = self.viz_engine.create_tech_breakdown_chart(weights, weighted_scores)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No detailed scoring breakdown available.")
        else:
            st.info("No component breakdown data available.")
    
    def _show_philosophical_insights(self, results: Dict) -> None:
        """Display philosophical insights"""
        philosophical_insights = results.get('philosophical_insights', [])
        cultural_insights = results.get('cultural_insights', {}).get('philosophical_insights', [])
        
        all_insights = philosophical_insights + cultural_insights
        
        if all_insights:
            st.markdown("""
            <div class="tech-panel">
                <h4 style="color: #00FFFF; margin-top: 0;">🧠 Philosophical Depth</h4>
            """, unsafe_allow_html=True)
            
            for insight in all_insights[:3]:
                st.markdown(f"• {insight}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Create philosophical depth chart
            depth_scores = {}
            for idx, insight in enumerate(all_insights[:5]):
                depth_scores[f"Insight {idx+1}"] = random.uniform(0.6, 0.95)
            
            if depth_scores:
                fig = self.viz_engine.create_philosophical_radial_chart(depth_scores)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No philosophical insights available for this analysis.")
    
    def _show_strengths_weaknesses(self, results: Dict) -> None:
        """Display strengths and weaknesses"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="grid-cell" style="background: rgba(0, 255, 100, 0.1);">
                <h4 style="color: #00FFAA; margin-top: 0;">✅ Strengths</h4>
            </div>
            """, unsafe_allow_html=True)
            
            strengths = results.get('strengths', [])
            if strengths:
                for strength in strengths:
                    st.markdown(f"• {strength}")
            else:
                st.info("No strengths identified.")
        
        with col2:
            st.markdown("""
            <div class="grid-cell" style="background: rgba(255, 100, 100, 0.1);">
                <h4 style="color: #FF6B6B; margin-top: 0;">⚠️ Areas for Improvement</h4>
            </div>
            """, unsafe_allow_html=True)
            
            weaknesses = results.get('weaknesses', [])
            if weaknesses:
                for weakness in weaknesses:
                    st.markdown(f"• {weakness}")
            else:
                st.info("No specific weaknesses identified.")
    
    def _show_recommendations(self, results: Dict) -> None:
        """Display recommendations"""
        recommendations = results.get('recommendations', [])
        ai_suggestions = results.get('ai_tool_suggestions', [])
        
        if recommendations:
            st.markdown("""
            <div class="grid-cell" style="background: rgba(0, 150, 255, 0.1);">
                <h4 style="color: #45B7D1; margin-top: 0;">💡 Recommendations</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
        
        if ai_suggestions:
            st.markdown("""
            <div class="grid-cell" style="background: rgba(150, 0, 255, 0.1); margin-top: 20px;">
                <h4 style="color: #764ba2; margin-top: 0;">🤖 AI Tool Suggestions</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for tool in ai_suggestions:
                st.markdown(f"**{tool.get('tool', 'Tool')}**: {tool.get('purpose', '')} - {tool.get('benefit', '')}")
    
    def _show_audience_festival(self, results: Dict) -> None:
        """Display audience and festival analysis"""
        audience = results.get('audience_analysis', {})
        festival = results.get('festival_recommendations', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            if audience:
                st.markdown("""
                <div class="grid-cell">
                    <h4 style="color: #FFD700; margin-top: 0;">🎯 Target Audience</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Primary Audiences:**")
                for aud in audience.get('target_audiences', [])[:3]:
                    st.markdown(f"• {aud}")
                
                st.metric("Engagement Score", f"{audience.get('engagement_score', 0):.2f}")
                st.metric("Market Potential", audience.get('market_potential', 'Unknown'))
        
        with col2:
            if festival:
                st.markdown("""
                <div class="grid-cell">
                    <h4 style="color: #FF8E53; margin-top: 0;">🎪 Festival Recommendations</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Festival Level", festival.get('level', 'Unknown').title())
                
                st.markdown("**Recommended Festivals:**")
                for fest in festival.get('festivals', [])[:3]:
                    st.markdown(f"• {fest}")
    
    def _show_ai_insights(self, results: Dict) -> None:
        """Display AI insights"""
        st.markdown("""
        <div class="tech-panel">
            <h4 style="color: #00FFFF; margin-top: 0;">🤖 AI Analysis Insights</h4>
        """, unsafe_allow_html=True)
        
        # Narrative analysis
        narrative = results.get('narrative_arc_analysis', {})
        if narrative:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", narrative.get('word_count', 0))
            with col2:
                st.metric("Lexical Diversity", f"{narrative.get('lexical_diversity', 0):.3f}")
            with col3:
                st.metric("Readability", f"{narrative.get('readability_score', 0):.2f}")
        
        # Character analysis
        characters = results.get('character_ecosystem', {})
        if characters:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Character Count", characters.get('potential_characters', 0))
            with col2:
                st.metric("Character Density", f"{characters.get('character_density', 0):.3f}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# DASHBOARD
# --------------------------
class EnhancedDashboard:
    """Enhanced dashboard with cinematic visualization and analytics"""
    
    def __init__(self, analysis_engine, persistence):
        self.engine = analysis_engine
        self.persistence = persistence
        self.viz_engine = CinematicVisualizationEngine()
    
    def show_dashboard(self) -> None:
        """Display the main dashboard"""
        st.title("🎬 FlickFinder AI - Cinematic Intelligence Dashboard")
        st.markdown("---")
        
        # Top stats row
        self._show_top_stats()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._show_main_content()
        
        with col2:
            self._show_sidebar_content()
    
    def _show_top_stats(self) -> None:
        """Display top statistics row"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Films Analyzed", st.session_state.analysis_count)
        
        with col2:
            if st.session_state.analysis_count > 0:
                history = self.persistence.get_all_history()
                scores = [h.get('overall_score', 0) for h in history]
                avg_score = np.mean(scores) if scores else 0
                st.metric("Average Score", f"{avg_score:.1f}")
            else:
                st.metric("Average Score", "0.0")
        
        with col3:
            if st.session_state.last_analysis_time:
                try:
                    last_time = datetime.fromisoformat(st.session_state.last_analysis_time)
                    time_ago = (datetime.now() - last_time).seconds // 60
                    st.metric("Last Analysis", f"{time_ago} min ago")
                except:
                    st.metric("Last Analysis", "Recently")
            else:
                st.metric("Last Analysis", "Never")
        
        with col4:
            elite_count = len([f for f in self.persistence.get_all_films() 
                             if f['analysis_results']['overall_score'] >= 4.0])
            st.metric("Elite Films", elite_count)
    
    def _show_main_content(self) -> None:
        """Display main content area"""
        # Quick analysis section
        st.markdown("### ⚡ Quick Film Analysis")
        
        tab1, tab2 = st.tabs(["📝 Manual Input", "🎥 YouTube URL"])
        
        with tab1:
            self._show_manual_input()
        
        with tab2:
            self._show_youtube_input()
        
        # Recent analyses
        if st.session_state.analysis_history:
            st.markdown("### 📅 Recent Analyses")
            recent_history = st.session_state.analysis_history[-3:]
            
            for idx, film in enumerate(reversed(recent_history)):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{film['title']}**")
                        st.caption(film.get('synopsis', '')[:100] + "..." if film.get('synopsis') else "No synopsis")
                    with col2:
                        score = film.get('overall_score', 0)
                        score_color = "#00FFFF" if score >= 4.0 else "#FFD700" if score >= 3.0 else "#AAAAAA"
                        st.markdown(f"<span style='color: {score_color}; font-size: 1.2em; font-weight: bold;'>{score:.1f}</span>", 
                                  unsafe_allow_html=True)
                    with col3:
                        if st.button("View", key=f"recent_{idx}"):
                            result = self.persistence.load_results(film['id'])
                            if result:
                                st.session_state.current_results_display = result['analysis_results']
                                st.session_state.show_results_page = True
                                st.session_state.current_page = "📊 Results"
                                st.rerun()
    
    def _show_sidebar_content(self) -> None:
        """Display sidebar content"""
        st.markdown("### 🚀 Quick Actions")
        
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("📊 View ePortfolio", use_container_width=True):
                st.session_state.current_page = "📂 ePortfolio"
                st.rerun()
            
            if st.button("🎯 Compare Films", use_container_width=True):
                st.session_state.current_page = "⚖️ Compare Films"
                st.rerun()
        
        with action_col2:
            if st.button("🔄 Clear History", use_container_width=True):
                self.persistence.clear_history()
                st.rerun()
            
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state.current_page = "⚙️ Settings"
                st.rerun()
        
        # Top films
        if st.session_state.top_films:
            st.markdown("### 🏆 Top Films")
            for idx, film in enumerate(st.session_state.top_films):
                film_data = film['film_data']
                analysis = film['analysis_results']
                
                st.markdown(f"""
                <div class="grid-cell" style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{film_data.get('title', 'Unknown')[:20]}</strong><br>
                            <span style="color: #AAAAAA; font-size: 0.8em;">
                                Score: <span style="color: #FFD700; font-weight: bold;">{analysis['overall_score']:.1f}</span>
                            </span>
                        </div>
                        <span style="background: rgba(0, 255, 255, 0.2); color: #00FFFF; 
                                  padding: 2px 8px; border-radius: 10px; font-size: 0.7em;">
                            #{idx + 1}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def _show_manual_input(self) -> None:
        """Show manual input form"""
        with st.form("manual_analysis_form"):
            title = st.text_input("Film Title", placeholder="Enter film title...")
            synopsis = st.text_area("Synopsis", placeholder="Enter film synopsis...", height=150)
            genre = st.selectbox("Genre (optional)", ["", "Drama", "Comedy", "Horror", "Sci-Fi", "Action", 
                                                     "Thriller", "Romance", "Documentary", "Fantasy", 
                                                     "Black Cinema", "Urban Drama"])
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("🚀 Analyze Film", use_container_width=True)
            with col2:
                st.form_submit_button("Clear", use_container_width=True)
            
            if submit and title and synopsis:
                with st.spinner("Analyzing film with cinematic intelligence..."):
                    film_data = {
                        'title': title,
                        'synopsis': synopsis,
                        'genre': genre,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    results = self.engine.analyze_film(film_data)
                    
                    if results:
                        st.success("Analysis complete!")
                        st.session_state.current_results_display = results
                        st.session_state.show_results_page = True
                        st.session_state.current_page = "📊 Results"
                        st.rerun()
    
    def _show_youtube_input(self) -> None:
        """Show YouTube input form"""
        youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        
        if youtube_url:
            video_id = TextProcessor.extract_video_id(youtube_url)
            
            if video_id:
                st.session_state.current_video_id = video_id
                
                # Try to get transcript
                transcript = TextProcessor.get_youtube_transcript(video_id)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📝 Analyze Transcript", use_container_width=True):
                        film_data = {
                            'title': f"YouTube Video: {video_id}",
                            'synopsis': "Analysis of YouTube video transcript",
                            'transcript': transcript or "",
                            'video_id': video_id,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        with st.spinner("Analyzing video content..."):
                            results = self.engine.analyze_film(film_data)
                            if results:
                                st.success("Video analysis complete!")
                                st.session_state.current_results_display = results
                                st.session_state.show_results_page = True
                                st.session_state.current_page = "📊 Results"
                                st.rerun()
                
                with col2:
                    if st.button("🎬 View Video", use_container_width=True):
                        st.session_state.current_page = "🎬 Video Viewer"
                        st.rerun()

# --------------------------
# VIDEO VIEWER
# --------------------------
class VideoViewer:
    """YouTube video viewer component"""
    
    @staticmethod
    def show_video_viewer() -> None:
        """Display YouTube video viewer"""
        st.title("🎬 Video Viewer")
        st.markdown("---")
        
        if not st.session_state.current_video_id:
            st.warning("No video selected. Please analyze a YouTube video first.")
            if st.button("← Back to Dashboard"):
                st.session_state.current_page = "🏠 Dashboard"
                st.rerun()
            return
        
        video_id = st.session_state.current_video_id
        
        # Video display
        st.markdown(f"""
        <div style="display: flex; justify-content: center;">
            <iframe width="800" height="450" 
                    src="https://www.youtube.com/embed/{video_id}" 
                    frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen>
            </iframe>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Show Analysis", use_container_width=True):
                if st.session_state.current_results_display:
                    st.session_state.current_page = "📊 Results"
                    st.rerun()
                else:
                    st.warning("No analysis available for this video.")
        
        with col2:
            if st.button("← Back to Dashboard", use_container_width=True):
                st.session_state.current_page = "🏠 Dashboard"
                st.rerun()

# --------------------------
# SETTINGS PAGE
# --------------------------
class SettingsPage:
    """Settings page component"""
    
    @staticmethod
    def show_settings() -> None:
        """Display settings page"""
        st.title("⚙️ Settings")
        st.markdown("---")
        
        # Display settings
        st.subheader("Display Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            holographic_mode = st.toggle("Holographic Mode", 
                                        value=st.session_state.get('holographic_mode', True))
            st.session_state.holographic_mode = holographic_mode
        
        with col2:
            tech_view = st.toggle("Technical View", 
                                 value=st.session_state.get('show_tech_view', False))
            st.session_state.show_tech_view = tech_view
        
        # Data management
        st.subheader("Data Management")
        
        if st.button("Clear All Data", type="secondary"):
            PersistenceManager.clear_history()
            st.success("All data cleared!")
            st.rerun()
        
        # Export data
        if st.session_state.analysis_history:
            if st.button("Export Analysis History"):
                df = pd.DataFrame(st.session_state.analysis_history)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"flickfinder_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Back button
        if st.button("← Back to Dashboard", use_container_width=True):
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()

# --------------------------
# MAIN APPLICATION
# --------------------------
class FlickFinderAI:
    """Main FlickFinder AI application"""
    
    def __init__(self):
        SessionManager.initialize()
        self.analysis_engine = FilmAnalysisEngine()
        self.persistence = PersistenceManager()
        self.viz_engine = CinematicVisualizationEngine()
        self.dashboard = EnhancedDashboard(self.analysis_engine, self.persistence)
        self.results_display = EnhancedResultsDisplay(self.viz_engine)
        self.eportfolio = EnhancedEPortfolioSystem(self.persistence, self.viz_engine)
        self.comparison_system = EnhancedComparisonSystem(self.persistence, self.viz_engine)
        self.video_viewer = VideoViewer()
        self.settings_page = SettingsPage()
    
    def run(self) -> None:
        """Main application loop"""
        # Sidebar navigation
        with st.sidebar:
            st.title("🎬 FlickFinder AI")
            st.markdown("---")
            
            # Navigation
            pages = {
                "🏠 Dashboard": "dashboard",
                "📊 Results": "results",
                "📂 ePortfolio": "eportfolio",
                "🎬 Video Viewer": "video_viewer",
                "⚖️ Compare Films": "compare",
                "⚙️ Settings": "settings"
            }
            
            # Ensure current_page exists in pages
            if st.session_state.current_page not in pages:
                st.session_state.current_page = "🏠 Dashboard"
            
            current_page = st.selectbox(
                "Navigate",
                list(pages.keys()),
                index=list(pages.keys()).index(st.session_state.current_page),
                key="nav_select"
            )
            
            st.session_state.current_page = current_page
            
            # Quick stats
            st.markdown("---")
            st.markdown("### 📈 Quick Stats")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Films", st.session_state.analysis_count)
            with col2:
                if st.session_state.analysis_count > 0:
                    history = self.persistence.get_all_history()
                    scores = [h.get('overall_score', 0) for h in history]
                    avg_score = np.mean(scores) if scores else 0
                    st.metric("Avg", f"{avg_score:.1f}")
            
            # Footer
            st.markdown("---")
            st.markdown("""
            <div style="text-align: center; color: #666;">
                <small>FlickFinder AI v4.0<br>Cinemative Intelligence</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Main content area
        try:
            if st.session_state.current_page == "🏠 Dashboard":
                if st.session_state.show_results_page and st.session_state.current_results_display:
                    self.results_display.show_results(st.session_state.current_results_display)
                else:
                    self.dashboard.show_dashboard()
            
            elif st.session_state.current_page == "📊 Results":
                if st.session_state.current_results_display:
                    self.results_display.show_results(st.session_state.current_results_display)
                else:
                    st.info("No results to display. Please analyze a film first.")
                    if st.button("← Back to Dashboard"):
                        st.session_state.current_page = "🏠 Dashboard"
                        st.rerun()
            
            elif st.session_state.current_page == "📂 ePortfolio":
                self.eportfolio.show_portfolio()
            
            elif st.session_state.current_page == "🎬 Video Viewer":
                self.video_viewer.show_video_viewer()
            
            elif st.session_state.current_page == "⚖️ Compare Films":
                self.comparison_system.show_comparison()
            
            elif st.session_state.current_page == "⚙️ Settings":
                self.settings_page.show_settings()
            
            else:
                # Default to dashboard
                self.dashboard.show_dashboard()
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.info("Returning to dashboard...")
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()

# --------------------------
# RUN APPLICATION
# --------------------------
if __name__ == "__main__":
    app = FlickFinderAI()
    app.run()
