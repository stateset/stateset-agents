"""
Multi-Objective Reward System for GRPO Agent Framework

This module provides sophisticated reward functions that combine multiple
evaluation criteria with weighted scoring, heuristic fallbacks, and 
domain-specific optimizations.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import re
    from collections import Counter
    import statistics
    ANALYSIS_LIBS_AVAILABLE = True
except ImportError:
    ANALYSIS_LIBS_AVAILABLE = False

from ..core.reward import RewardFunction, RewardResult
from ..core.trajectory import ConversationTurn
# from ..utils.cache import CacheService  # TODO: Implement cache service
# from ..utils.monitoring import MonitoringService  # TODO: Implement monitoring service

logger = logging.getLogger(__name__)


@dataclass
class RewardComponent:
    """Individual reward component with weight and configuration"""
    name: str
    weight: float
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")


class BaseRewardComponent(ABC):
    """Base class for individual reward components"""
    
    def __init__(self, name: str, weight: float = 1.0, **kwargs):
        self.name = name
        self.weight = weight
        self.config = kwargs
        
    @abstractmethod
    async def compute_score(
        self,
        turns: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute score for this component (0-1 range)"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {
            "name": self.name,
            "weight": self.weight,
            "config": self.config
        }


class LengthRewardComponent(BaseRewardComponent):
    """Reward component based on response length"""
    
    def __init__(
        self,
        name: str = "length",
        weight: float = 0.1,
        min_length: int = 10,
        max_length: int = 200,
        optimal_range: Tuple[int, int] = (50, 150),
        **kwargs
    ):
        super().__init__(name, weight, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.optimal_range = optimal_range
    
    async def compute_score(
        self,
        turns: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on response length"""
        if not turns:
            return 0.0
        
        # Get last assistant response
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0
        
        last_response = assistant_responses[-1].get("content", "")
        word_count = len(last_response.split())
        
        # Penalty for too short or too long
        if word_count < self.min_length:
            return 0.2
        if word_count > self.max_length:
            return 0.3
        
        # Reward optimal range
        if self.optimal_range[0] <= word_count <= self.optimal_range[1]:
            return 1.0
        
        # Gradual penalty outside optimal range
        if word_count < self.optimal_range[0]:
            return 0.5 + 0.5 * (word_count - self.min_length) / (self.optimal_range[0] - self.min_length)
        else:
            return 0.5 + 0.5 * (self.max_length - word_count) / (self.max_length - self.optimal_range[1])


class EmpathyRewardComponent(BaseRewardComponent):
    """Reward component based on empathy indicators with sentiment analysis"""
    
    def __init__(
        self,
        name: str = "empathy",
        weight: float = 0.2,
        empathy_keywords: Optional[List[str]] = None,
        use_sentiment_analysis: bool = True,
        **kwargs
    ):
        super().__init__(name, weight, **kwargs)
        self.empathy_keywords = empathy_keywords or [
            "understand", "sorry", "apologize", "feel", "frustrat", "concern",
            "worry", "help", "support", "appreciate", "thank", "welcome",
            "please", "certainly", "absolutely", "definitely", "of course"
        ]
        self.use_sentiment_analysis = use_sentiment_analysis and (TEXTBLOB_AVAILABLE or VADER_AVAILABLE)
        
        # Initialize sentiment analyzers if available
        self.vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # Emotional intensity keywords
        self.high_empathy_phrases = [
            "i understand how", "that must be", "i can imagine", "i'm so sorry",
            "that sounds", "i feel for you", "i completely understand",
            "i know this is", "that's really", "i hear you"
        ]
        
        self.supportive_phrases = [
            "let me help", "i'll take care", "don't worry", "we'll figure",
            "i'm here to", "let's work", "together we", "i'll make sure"
        ]
    
    async def compute_score(
        self,
        turns: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on empathy indicators with sentiment analysis"""
        if not turns:
            return 0.0
        
        # Get user and assistant responses
        user_messages = [t for t in turns if t.get("role") == "user"]
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        
        if not assistant_responses:
            return 0.0
        
        # Analyze user sentiment to understand emotional context
        user_emotional_state = self._analyze_user_sentiment(user_messages)
        
        # Analyze assistant empathy response
        empathy_score = await self._analyze_empathy_response(assistant_responses, user_emotional_state)
        
        return empathy_score
    
    def _analyze_user_sentiment(self, user_messages: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze user's emotional state from their messages"""
        if not user_messages or not self.use_sentiment_analysis:
            return {"negative": 0.0, "neutral": 0.5, "positive": 0.0, "compound": 0.0}
        
        # Combine all user messages
        user_text = " ".join([msg.get("content", "") for msg in user_messages])
        
        sentiment_scores = {"negative": 0.0, "neutral": 0.5, "positive": 0.0, "compound": 0.0}
        
        # VADER sentiment analysis (better for social media text)
        if self.vader_analyzer:
            vader_scores = self.vader_analyzer.polarity_scores(user_text)
            sentiment_scores.update(vader_scores)
        
        # TextBlob sentiment analysis
        elif TEXTBLOB_AVAILABLE:
            blob = TextBlob(user_text)
            sentiment_scores["compound"] = blob.sentiment.polarity
            sentiment_scores["positive"] = max(0, blob.sentiment.polarity)
            sentiment_scores["negative"] = abs(min(0, blob.sentiment.polarity))
            sentiment_scores["neutral"] = 1 - abs(blob.sentiment.polarity)
        
        return sentiment_scores
    
    async def _analyze_empathy_response(
        self, 
        assistant_responses: List[Dict[str, Any]], 
        user_sentiment: Dict[str, float]
    ) -> float:
        """Analyze empathy in assistant responses based on user emotional state"""
        
        empathy_score = 0.0
        total_responses = len(assistant_responses)
        
        # Determine if user is in distress
        user_distress_level = user_sentiment.get("negative", 0.0)
        user_is_distressed = user_distress_level > 0.3
        
        for response in assistant_responses:
            content = response.get("content", "").lower()
            response_score = 0.0
            
            # Base empathy keyword matching
            keyword_matches = sum(1 for keyword in self.empathy_keywords if keyword in content)
            response_score += min(0.3, keyword_matches * 0.05)
            
            # High empathy phrases (stronger indicators)
            high_empathy_matches = sum(1 for phrase in self.high_empathy_phrases if phrase in content)
            response_score += min(0.4, high_empathy_matches * 0.2)
            
            # Supportive phrases
            supportive_matches = sum(1 for phrase in self.supportive_phrases if phrase in content)
            response_score += min(0.3, supportive_matches * 0.15)
            
            # Analyze response sentiment to match user's emotional state
            if self.use_sentiment_analysis:
                response_sentiment = self._get_response_sentiment(content)
                
                # If user is distressed, reward compassionate/understanding responses
                if user_is_distressed:
                    if response_sentiment.get("compound", 0) > 0.1:  # Positive but not overly cheerful
                        response_score += 0.2
                    elif response_sentiment.get("neutral", 0) > 0.5:  # Calm, understanding tone
                        response_score += 0.3
                else:
                    # For neutral/positive users, positive responses are good
                    if response_sentiment.get("compound", 0) > 0.2:
                        response_score += 0.2
            
            # Check for dismissive language (penalty)
            dismissive_phrases = ["just", "simply", "only need to", "all you have to do"]
            if user_is_distressed and any(phrase in content for phrase in dismissive_phrases):
                response_score -= 0.2
            
            # Personal pronouns indicating engagement
            personal_engagement = len(re.findall(r'\b(you|your|we|us|together)\b', content))
            response_score += min(0.1, personal_engagement * 0.02)
            
            empathy_score += response_score
        
        # Normalize by number of responses and ensure [0, 1] range
        final_score = empathy_score / max(1, total_responses)
        return max(0.0, min(1.0, final_score))
    
    def _get_response_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment scores for a response"""
        if self.vader_analyzer:
            return self.vader_analyzer.polarity_scores(text)
        elif TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            return {
                "compound": polarity,
                "positive": max(0, polarity),
                "negative": abs(min(0, polarity)),
                "neutral": 1 - abs(polarity)
            }
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}


class ActionOrientedRewardComponent(BaseRewardComponent):
    """Reward component based on action-oriented language"""
    
    def __init__(
        self,
        name: str = "action_oriented",
        weight: float = 0.2,
        action_keywords: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(name, weight, **kwargs)
        self.action_keywords = action_keywords or [
            "will", "can", "let me", "i'll", "here's", "you can", "try",
            "suggest", "recommend", "solution", "step", "first", "then",
            "next", "process", "procedure", "method", "approach", "way"
        ]
    
    async def compute_score(
        self,
        turns: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on action-oriented language"""
        if not turns:
            return 0.0
        
        # Get assistant responses
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0
        
        # Check for action keywords
        total_matches = 0
        
        for response in assistant_responses:
            content = response.get("content", "").lower()
            
            for keyword in self.action_keywords:
                if keyword in content:
                    total_matches += 1
        
        # Normalize by number of responses
        action_density = total_matches / max(1, len(assistant_responses))
        return min(1.0, action_density / 3.0)  # Scale to 0-1


class SimilarityRewardComponent(BaseRewardComponent):
    """Reward component based on similarity to expected responses"""
    
    def __init__(
        self,
        name: str = "similarity",
        weight: float = 0.3,
        expected_responses: Optional[List[str]] = None,
        similarity_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__(name, weight, **kwargs)
        self.expected_responses = expected_responses or []
        self.similarity_threshold = similarity_threshold
    
    async def compute_score(
        self,
        turns: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on similarity to expected responses"""
        if not turns or not self.expected_responses:
            return 0.5  # Neutral score if no expected responses
        
        # Get assistant responses
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0
        
        # Calculate similarity using simple keyword overlap
        max_similarity = 0.0
        
        for response in assistant_responses:
            content = response.get("content", "").lower()
            response_words = set(content.split())
            
            for expected in self.expected_responses:
                expected_words = set(expected.lower().split())
                
                if not expected_words:
                    continue
                
                # Jaccard similarity
                intersection = response_words & expected_words
                union = response_words | expected_words
                
                if union:
                    similarity = len(intersection) / len(union)
                    max_similarity = max(max_similarity, similarity)
        
        # Apply threshold and scale
        if max_similarity >= self.similarity_threshold:
            return min(1.0, max_similarity / self.similarity_threshold)
        else:
            return max_similarity / self.similarity_threshold * 0.5


class ProfessionalismRewardComponent(BaseRewardComponent):
    """Reward component based on professionalism indicators with sentiment analysis"""
    
    def __init__(
        self,
        name: str = "professionalism",
        weight: float = 0.2,
        professional_indicators: Optional[List[str]] = None,
        unprofessional_indicators: Optional[List[str]] = None,
        use_sentiment_analysis: bool = True,
        **kwargs
    ):
        super().__init__(name, weight, **kwargs)
        self.professional_indicators = professional_indicators or [
            "please", "thank you", "may i", "would you", "could you",
            "i'd be happy", "certainly", "absolutely", "professional",
            "assistance", "service", "support", "help", "resolve"
        ]
        self.unprofessional_indicators = unprofessional_indicators or [
            "whatever", "dunno", "idk", "lol", "omg", "wtf", "stupid",
            "dumb", "sucks", "hate", "annoying", "frustrated", "angry"
        ]
        self.use_sentiment_analysis = use_sentiment_analysis and (TEXTBLOB_AVAILABLE or VADER_AVAILABLE)
        
        # Initialize sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # Professional language patterns
        self.formal_phrases = [
            "i would be happy to", "i'd be pleased to", "it would be my pleasure",
            "i understand your concern", "let me assist you", "i'll be glad to help"
        ]
        
        self.business_language = [
            "regarding", "concerning", "furthermore", "additionally", "however",
            "therefore", "consequently", "nevertheless", "appropriate", "efficient"
        ]
    
    async def compute_score(
        self,
        turns: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on professionalism with sentiment analysis"""
        if not turns:
            return 0.0
        
        # Get assistant responses
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0
        
        total_score = 0.0
        
        for response in assistant_responses:
            content = response.get("content", "")
            response_score = await self._analyze_professionalism(content)
            total_score += response_score
        
        # Average across all responses
        return total_score / len(assistant_responses)
    
    async def _analyze_professionalism(self, content: str) -> float:
        """Analyze professionalism of a single response"""
        score = 0.5  # Base professional score
        content_lower = content.lower()
        
        # Professional indicators
        professional_count = sum(1 for indicator in self.professional_indicators 
                                if indicator in content_lower)
        score += min(0.25, professional_count * 0.05)
        
        # Formal phrases (higher weight)
        formal_count = sum(1 for phrase in self.formal_phrases 
                          if phrase in content_lower)
        score += min(0.15, formal_count * 0.1)
        
        # Business language
        business_count = sum(1 for term in self.business_language 
                           if term in content_lower)
        score += min(0.1, business_count * 0.02)
        
        # Unprofessional penalties
        unprofessional_count = sum(1 for indicator in self.unprofessional_indicators 
                                 if indicator in content_lower)
        score -= unprofessional_count * 0.1
        
        # Sentiment analysis for tone professionalism
        if self.use_sentiment_analysis:
            sentiment_score = self._analyze_professional_tone(content)
            score += sentiment_score
        
        # Grammar and structure analysis
        structure_score = self._analyze_structure(content)
        score += structure_score
        
        return max(0.0, min(1.0, score))
    
    def _analyze_professional_tone(self, content: str) -> float:
        """Analyze tone professionalism using sentiment"""
        if not self.use_sentiment_analysis:
            return 0.0
        
        tone_score = 0.0
        
        if self.vader_analyzer:
            sentiment = self.vader_analyzer.polarity_scores(content)
            
            # Professional tone should be neutral to slightly positive
            compound = sentiment.get('compound', 0)
            
            if 0.1 <= compound <= 0.6:  # Appropriately positive
                tone_score += 0.1
            elif -0.1 <= compound <= 0.1:  # Professional neutral
                tone_score += 0.08
            elif compound > 0.6:  # Too enthusiastic
                tone_score -= 0.05
            elif compound < -0.3:  # Too negative
                tone_score -= 0.1
        
        elif TEXTBLOB_AVAILABLE:
            blob = TextBlob(content)
            polarity = blob.sentiment.polarity
            
            if 0.1 <= polarity <= 0.5:  # Appropriately positive
                tone_score += 0.1
            elif -0.1 <= polarity <= 0.1:  # Professional neutral
                tone_score += 0.08
            elif polarity > 0.5:  # Too enthusiastic
                tone_score -= 0.05
            elif polarity < -0.3:  # Too negative
                tone_score -= 0.1
        
        return tone_score
    
    def _analyze_structure(self, content: str) -> float:
        """Analyze structural professionalism"""
        if not ANALYSIS_LIBS_AVAILABLE:
            return 0.0
        
        structure_score = 0.0
        
        # Sentence length analysis (professional responses have moderate sentence length)
        sentences = re.split(r'[.!?]+', content)
        if sentences:
            avg_sentence_length = statistics.mean([len(s.split()) for s in sentences if s.strip()])
            
            if 8 <= avg_sentence_length <= 20:  # Optimal range
                structure_score += 0.05
            elif avg_sentence_length < 5 or avg_sentence_length > 30:  # Too short or long
                structure_score -= 0.03
        
        # Capitalization (proper capitalization is professional)
        if content and content[0].isupper():
            structure_score += 0.02
        
        # Punctuation analysis
        if content.endswith(('.', '!', '?')):
            structure_score += 0.02
        
        # Check for excessive punctuation (unprofessional)
        excessive_punct = len(re.findall(r'[!]{2,}|[?]{2,}|[.]{3,}', content))
        if excessive_punct > 0:
            structure_score -= 0.05
        
        return structure_score


class SentimentAwarenessComponent(BaseRewardComponent):
    """Reward component that analyzes emotional intelligence and sentiment appropriateness"""
    
    def __init__(
        self,
        name: str = "sentiment_awareness",
        weight: float = 0.15,
        **kwargs
    ):
        super().__init__(name, weight, **kwargs)
        self.vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # Emotion detection keywords
        self.emotion_keywords = {
            'frustration': ['frustrated', 'annoying', 'irritating', 'upset', 'angry'],
            'confusion': ['confused', 'unclear', 'don\'t understand', 'puzzled'],
            'urgency': ['urgent', 'asap', 'immediately', 'quickly', 'emergency'],
            'satisfaction': ['great', 'excellent', 'perfect', 'wonderful', 'amazing'],
            'disappointment': ['disappointed', 'expected', 'hoped', 'let down']
        }
    
    async def compute_score(
        self,
        turns: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on sentiment awareness and emotional intelligence"""
        if not turns or not (TEXTBLOB_AVAILABLE or VADER_AVAILABLE):
            return 0.5  # Neutral score if sentiment analysis unavailable
        
        user_messages = [t for t in turns if t.get("role") == "user"]
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        
        if not user_messages or not assistant_responses:
            return 0.5
        
        # Analyze user emotional journey
        user_emotions = self._track_user_emotions(user_messages)
        
        # Analyze assistant's emotional responsiveness
        response_quality = await self._analyze_emotional_responsiveness(
            assistant_responses, user_emotions
        )
        
        return response_quality
    
    def _track_user_emotions(self, user_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Track user's emotional state throughout conversation"""
        emotions = []
        
        for message in user_messages:
            content = message.get("content", "")
            emotion_data = {
                "content": content,
                "sentiment": self._get_sentiment(content),
                "detected_emotions": self._detect_emotions(content),
                "intensity": self._calculate_emotional_intensity(content)
            }
            emotions.append(emotion_data)
        
        return emotions
    
    def _get_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment scores for text"""
        if self.vader_analyzer:
            return self.vader_analyzer.polarity_scores(text)
        elif TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            return {
                "compound": polarity,
                "positive": max(0, polarity),
                "negative": abs(min(0, polarity)),
                "neutral": 1 - abs(polarity)
            }
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
    
    def _detect_emotions(self, text: str) -> List[str]:
        """Detect specific emotions in text"""
        detected = []
        text_lower = text.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.append(emotion)
        
        return detected
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        """Calculate emotional intensity based on various factors"""
        intensity = 0.0
        
        # Exclamation marks
        intensity += min(0.3, text.count('!') * 0.1)
        
        # Question marks (confusion/urgency)
        intensity += min(0.2, text.count('?') * 0.05)
        
        # Capitalization (shouting)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        if caps_ratio > 0.3:
            intensity += min(0.4, caps_ratio)
        
        # Repetitive punctuation
        repetitive_punct = len(re.findall(r'[!]{2,}|[?]{2,}|[.]{3,}', text))
        intensity += min(0.3, repetitive_punct * 0.15)
        
        return min(1.0, intensity)
    
    async def _analyze_emotional_responsiveness(
        self, 
        responses: List[Dict[str, Any]], 
        user_emotions: List[Dict[str, Any]]
    ) -> float:
        """Analyze how well assistant responds to user emotions"""
        
        if not user_emotions:
            return 0.5
        
        total_score = 0.0
        scored_responses = 0
        
        for i, response in enumerate(responses):
            # Match response to user emotion (current or previous)
            relevant_user_emotion = user_emotions[min(i, len(user_emotions) - 1)]
            
            response_score = self._score_emotional_response(
                response.get("content", ""), 
                relevant_user_emotion
            )
            
            total_score += response_score
            scored_responses += 1
        
        return total_score / max(1, scored_responses)
    
    def _score_emotional_response(self, response_content: str, user_emotion: Dict[str, Any]) -> float:
        """Score a single response against user's emotional state"""
        score = 0.5  # Base score
        
        user_sentiment = user_emotion.get("sentiment", {})
        user_emotions_detected = user_emotion.get("detected_emotions", [])
        user_intensity = user_emotion.get("intensity", 0.0)
        
        response_sentiment = self._get_sentiment(response_content)
        
        # Score based on appropriate emotional response
        if user_sentiment.get("negative", 0) > 0.5:  # User is negative
            # Response should be empathetic but not overly positive
            if 0.0 <= response_sentiment.get("compound", 0) <= 0.3:
                score += 0.3  # Appropriate supportive tone
            elif response_sentiment.get("compound", 0) > 0.5:
                score -= 0.2  # Too cheerful for negative user
            
            # Check for empathetic language
            empathetic_words = ["understand", "sorry", "help", "support"]
            if any(word in response_content.lower() for word in empathetic_words):
                score += 0.2
        
        elif user_sentiment.get("positive", 0) > 0.5:  # User is positive
            # Response can be positive but professional
            if response_sentiment.get("compound", 0) > 0.1:
                score += 0.2
        
        # Handle specific emotions
        if "frustration" in user_emotions_detected:
            calming_words = ["understand", "help", "resolve", "fix", "solution"]
            if any(word in response_content.lower() for word in calming_words):
                score += 0.25
            
            # Avoid dismissive language
            dismissive = ["just", "simply", "only"]
            if any(word in response_content.lower() for word in dismissive):
                score -= 0.2
        
        if "urgency" in user_emotions_detected:
            urgent_response = ["immediately", "right away", "asap", "quickly", "priority"]
            if any(word in response_content.lower() for word in urgent_response):
                score += 0.2
        
        if "confusion" in user_emotions_detected:
            clarifying_words = ["explain", "clarify", "help you understand", "let me break"]
            if any(word in response_content.lower() for word in clarifying_words):
                score += 0.25
        
        # Adjust for emotional intensity
        if user_intensity > 0.7:  # High emotional intensity
            # Response should acknowledge the intensity
            acknowledging_phrases = ["i understand this is", "i know this", "this must be"]
            if any(phrase in response_content.lower() for phrase in acknowledging_phrases):
                score += 0.15
        
        return max(0.0, min(1.0, score))


class MultiObjectiveRewardFunction(RewardFunction):
    """
    Multi-objective reward function that combines multiple reward components
    with weighted scoring and sophisticated evaluation criteria.
    """
    
    def __init__(
        self,
        components: Optional[List[BaseRewardComponent]] = None,
        weight: float = 1.0,
        normalization_method: str = "weighted_sum",
        cache_service: Optional[Any] = None,  # TODO: Type with CacheService when implemented
        monitoring_service: Optional[Any] = None  # TODO: Type with MonitoringService when implemented
    ):
        """
        Initialize multi-objective reward function.
        
        Args:
            components: List of reward components
            weight: Overall weight for this reward function
            normalization_method: How to combine component scores
            cache_service: Optional cache service
            monitoring_service: Optional monitoring service
        """
        super().__init__(weight=weight)
        
        self.components = components or []
        self.normalization_method = normalization_method
        self.cache = cache_service
        self.monitoring = monitoring_service
        
        # Validate weights sum to reasonable range
        total_weight = sum(c.weight for c in self.components)
        if total_weight > 0:
            # Normalize weights to sum to 1
            for component in self.components:
                component.weight = component.weight / total_weight
        
        # Metrics
        self.evaluation_count = 0
        self.component_scores = {}
        
    def add_component(self, component: BaseRewardComponent):
        """Add a reward component"""
        self.components.append(component)
        
        # Re-normalize weights
        total_weight = sum(c.weight for c in self.components)
        if total_weight > 0:
            for comp in self.components:
                comp.weight = comp.weight / total_weight
    
    def remove_component(self, name: str):
        """Remove a reward component by name"""
        self.components = [c for c in self.components if c.name != name]
        
        # Re-normalize weights
        total_weight = sum(c.weight for c in self.components)
        if total_weight > 0:
            for comp in self.components:
                comp.weight = comp.weight / total_weight
    
    async def compute_reward(
        self,
        turns: List[ConversationTurn],
        context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """
        Compute multi-objective reward.
        
        Args:
            turns: List of conversation turns
            context: Optional context information
            
        Returns:
            RewardResult with combined score and component breakdown
        """
        # Convert ConversationTurn objects to dict format for backward compatibility
        turns_dict = [turn.to_dict() if hasattr(turn, 'to_dict') else turn for turn in turns]
        
        if not self.components:
            return RewardResult(score=0.0, breakdown={"error": "No components configured"}, metadata={})
        
        # Compute scores for each component
        component_scores = {}
        weighted_scores = []
        
        for component in self.components:
            try:
                score = await component.compute_score(turns_dict, context)
                component_scores[component.name] = score
                weighted_scores.append(score * component.weight)
                
                # Track component performance
                if component.name not in self.component_scores:
                    self.component_scores[component.name] = []
                self.component_scores[component.name].append(score)
                
            except Exception as e:
                logger.error(f"Component {component.name} failed: {e}")
                component_scores[component.name] = 0.0
                weighted_scores.append(0.0)
        
        # Combine scores
        if self.normalization_method == "weighted_sum":
            final_score = sum(weighted_scores)
        elif self.normalization_method == "weighted_average":
            final_score = sum(weighted_scores) / len(weighted_scores)
        elif self.normalization_method == "geometric_mean":
            # Geometric mean of weighted scores
            if all(s > 0 for s in weighted_scores):
                final_score = np.prod(weighted_scores) ** (1.0 / len(weighted_scores))
            else:
                final_score = 0.0
        else:
            final_score = sum(weighted_scores)
        
        # Ensure score is in [0, 1]
        final_score = max(0.0, min(1.0, final_score))
        
        self.evaluation_count += 1
        
        # Create breakdown
        breakdown = {
            "total_score": final_score,
            "components": component_scores,
            "weights": {c.name: c.weight for c in self.components},
            "normalization_method": self.normalization_method,
            "evaluation_count": self.evaluation_count
        }
        
        # Log metrics
        if self.monitoring:
            await self.monitoring.log_metric(
                "multi_objective_reward.evaluation",
                final_score,
                tags={"method": self.normalization_method}
            )
        
        return RewardResult(
            score=final_score, 
            breakdown=breakdown,
            metadata={"evaluation_timestamp": datetime.now().isoformat()}
        )
    
    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each component"""
        stats = {}
        
        for name, scores in self.component_scores.items():
            if scores:
                stats[name] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "count": len(scores)
                }
        
        return stats
    
    def get_info(self) -> Dict[str, Any]:
        """Get reward function information"""
        return {
            "type": "multi_objective",
            "components": [c.get_info() for c in self.components],
            "normalization_method": self.normalization_method,
            "evaluation_count": self.evaluation_count,
            "component_statistics": self.get_component_statistics()
        }


# Convenience functions for creating common multi-objective rewards
def create_customer_service_reward(
    expected_responses: Optional[List[str]] = None,
    weight: float = 1.0,
    use_sentiment_analysis: bool = True,
    **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for customer service with sentiment analysis"""
    components = [
        EmpathyRewardComponent(weight=0.25, use_sentiment_analysis=use_sentiment_analysis),
        ActionOrientedRewardComponent(weight=0.2),
        ProfessionalismRewardComponent(weight=0.2, use_sentiment_analysis=use_sentiment_analysis),
        LengthRewardComponent(weight=0.1, min_length=20, optimal_range=(50, 200)),
        SimilarityRewardComponent(weight=0.1, expected_responses=expected_responses)
    ]
    
    # Add sentiment awareness if available
    if use_sentiment_analysis and (TEXTBLOB_AVAILABLE or VADER_AVAILABLE):
        components.append(SentimentAwarenessComponent(weight=0.15))
    
    return MultiObjectiveRewardFunction(
        components=components,
        weight=weight,
        **kwargs
    )


def create_technical_support_reward(
    expected_responses: Optional[List[str]] = None,
    weight: float = 1.0,
    use_sentiment_analysis: bool = True,
    **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for technical support with sentiment analysis"""
    components = [
        ActionOrientedRewardComponent(weight=0.35),  # High weight for solutions
        ProfessionalismRewardComponent(weight=0.2, use_sentiment_analysis=use_sentiment_analysis),
        LengthRewardComponent(weight=0.1, min_length=30, optimal_range=(100, 300)),
        SimilarityRewardComponent(weight=0.2, expected_responses=expected_responses)
    ]
    
    # Add sentiment awareness for technical support
    if use_sentiment_analysis and (TEXTBLOB_AVAILABLE or VADER_AVAILABLE):
        components.append(SentimentAwarenessComponent(weight=0.15))
    
    return MultiObjectiveRewardFunction(
        components=components,
        weight=weight,
        **kwargs
    )


def create_educational_reward(
    expected_responses: Optional[List[str]] = None,
    weight: float = 1.0,
    **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for educational content"""
    components = [
        ActionOrientedRewardComponent(weight=0.3),
        ProfessionalismRewardComponent(weight=0.2),
        LengthRewardComponent(weight=0.1, min_length=50, optimal_range=(150, 400)),
        SimilarityRewardComponent(weight=0.4, expected_responses=expected_responses)
    ]
    
    return MultiObjectiveRewardFunction(
        components=components,
        weight=weight,
        **kwargs
    )


def create_sales_reward(
    expected_responses: Optional[List[str]] = None,
    weight: float = 1.0,
    **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for sales interactions"""
    # Add sales-specific components
    sales_action_keywords = [
        "buy", "purchase", "order", "discount", "offer", "deal", "save",
        "recommend", "suggest", "perfect for", "ideal", "best choice",
        "limited time", "special", "exclusive", "value", "benefit"
    ]
    
    components = [
        ActionOrientedRewardComponent(weight=0.35, action_keywords=sales_action_keywords),
        ProfessionalismRewardComponent(weight=0.25),
        EmpathyRewardComponent(weight=0.15),
        LengthRewardComponent(weight=0.1, min_length=30, optimal_range=(80, 250)),
        SimilarityRewardComponent(weight=0.15, expected_responses=expected_responses)
    ]
    
    return MultiObjectiveRewardFunction(
        components=components,
        weight=weight,
        **kwargs
    )


def create_creative_reward(
    expected_responses: Optional[List[str]] = None,
    weight: float = 1.0,
    **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for creative content"""
    creative_keywords = [
        "imagine", "creative", "unique", "original", "innovative", "artistic",
        "beautiful", "inspiring", "fascinating", "amazing", "wonderful",
        "story", "narrative", "character", "scene", "vivid", "descriptive"
    ]
    
    components = [
        ActionOrientedRewardComponent(weight=0.2, action_keywords=creative_keywords),
        LengthRewardComponent(weight=0.2, min_length=50, optimal_range=(200, 500)),
        SimilarityRewardComponent(weight=0.6, expected_responses=expected_responses)
    ]
    
    return MultiObjectiveRewardFunction(
        components=components,
        weight=weight,
        **kwargs
    )