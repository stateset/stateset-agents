"""
Enhanced Multi-Objective Reward System with Sentiment Analysis - Usage Example

This example demonstrates how to use the enhanced reward system with sentiment analysis
for customer service scenarios.
"""

import asyncio
from rewards.multi_objective_reward import (
    create_customer_service_reward,
    create_technical_support_reward,
    EmpathyRewardComponent,
    SentimentAwarenessComponent,
    MultiObjectiveRewardFunction
)
from core.trajectory import ConversationTurn
from datetime import datetime


async def test_sentiment_aware_rewards():
    """Test the enhanced sentiment-aware reward system"""
    
    # Example conversation turns
    frustrated_user_scenario = [
        ConversationTurn(
            role="user",
            content="I'm really frustrated! My order was supposed to arrive 3 days ago and I still haven't received it! This is completely unacceptable!!!"
        ),
        ConversationTurn(
            role="assistant", 
            content="I completely understand how frustrating this must be for you, and I sincerely apologize for the delay with your order. Let me immediately check the tracking information and see what I can do to resolve this situation for you right away."
        ),
        ConversationTurn(
            role="user",
            content="Thank you, I really need this resolved quickly as it's a gift for someone."
        ),
        ConversationTurn(
            role="assistant",
            content="I can see the urgency of this situation. I've located your order and I'm expediting a replacement shipment that will arrive tomorrow with priority delivery. I'll also apply a full refund for the shipping costs and include a 20% discount on your next order for the inconvenience."
        )
    ]
    
    confused_user_scenario = [
        ConversationTurn(
            role="user",
            content="I'm confused about how to set up this software. The instructions don't make sense to me."
        ),
        ConversationTurn(
            role="assistant",
            content="I understand that the setup process can be confusing. Let me break this down into simple steps and walk you through each one clearly. First, let's start with downloading the installer from our website."
        ),
        ConversationTurn(
            role="user",
            content="Okay, that's clearer. What's the next step?"
        ),
        ConversationTurn(
            role="assistant",
            content="Great! Now that you have the installer, please run it by double-clicking the file. You'll see a setup wizard that will guide you through the process. Would you like me to explain what each screen will show you?"
        )
    ]
    
    # Create sentiment-aware customer service reward
    reward_function = create_customer_service_reward(
        expected_responses=[
            "I understand your frustration",
            "Let me help you with that",
            "I'll resolve this right away"
        ],
        use_sentiment_analysis=True
    )
    
    print("Testing Enhanced Multi-Objective Reward System with Sentiment Analysis")
    print("=" * 70)
    
    # Test frustrated user scenario
    print("\n1. Frustrated User Scenario:")
    print("-" * 30)
    
    result1 = await reward_function.compute_reward(frustrated_user_scenario)
    print(f"Overall Score: {result1.score:.3f}")
    print("Component Breakdown:")
    for component, score in result1.breakdown.get("components", {}).items():
        print(f"  - {component}: {score:.3f}")
    
    # Test confused user scenario  
    print("\n2. Confused User Scenario:")
    print("-" * 30)
    
    result2 = await reward_function.compute_reward(confused_user_scenario)
    print(f"Overall Score: {result2.score:.3f}")
    print("Component Breakdown:")
    for component, score in result2.breakdown.get("components", {}).items():
        print(f"  - {component}: {score:.3f}")
    
    # Test individual sentiment awareness component
    print("\n3. Sentiment Awareness Component Analysis:")
    print("-" * 45)
    
    sentiment_component = SentimentAwarenessComponent()
    
    # Convert ConversationTurn to dict format for component
    frustrated_scenario_dicts = [turn.to_dict() for turn in frustrated_user_scenario]
    sentiment_score = await sentiment_component.compute_score(frustrated_scenario_dicts)
    
    print(f"Sentiment Awareness Score for Frustrated User: {sentiment_score:.3f}")
    
    confused_scenario_dicts = [turn.to_dict() for turn in confused_user_scenario]
    sentiment_score2 = await sentiment_component.compute_score(confused_scenario_dicts)
    
    print(f"Sentiment Awareness Score for Confused User: {sentiment_score2:.3f}")
    
    # Test technical support scenario
    print("\n4. Technical Support Scenario:")
    print("-" * 35)
    
    tech_reward = create_technical_support_reward(use_sentiment_analysis=True)
    
    tech_scenario = [
        ConversationTurn(
            role="user",
            content="My computer keeps crashing when I try to run this application. It's really urgent as I have a deadline tomorrow!"
        ),
        ConversationTurn(
            role="assistant",
            content="I understand this is urgent and I'll help you resolve this crash issue immediately. Let's start by checking your system requirements and then run some diagnostic steps to identify the root cause."
        )
    ]
    
    tech_result = await tech_reward.compute_reward(tech_scenario)
    print(f"Technical Support Score: {tech_result.score:.3f}")
    print("Component Breakdown:")
    for component, score in tech_result.breakdown.get("components", {}).items():
        print(f"  - {component}: {score:.3f}")
    
    # Performance statistics
    print("\n5. Component Performance Statistics:")
    print("-" * 40)
    
    stats = reward_function.get_component_statistics()
    for component_name, component_stats in stats.items():
        print(f"\n{component_name}:")
        for stat_name, stat_value in component_stats.items():
            print(f"  {stat_name}: {stat_value:.3f}")


async def test_empathy_component_enhancements():
    """Test the enhanced empathy component with sentiment analysis"""
    
    print("\n6. Enhanced Empathy Component Testing:")
    print("-" * 45)
    
    empathy_component = EmpathyRewardComponent(use_sentiment_analysis=True)
    
    # Test different emotional scenarios
    scenarios = [
        {
            "name": "Highly Distressed User",
            "turns": [
                {"role": "user", "content": "I'm absolutely devastated! My wedding photos are completely gone from your cloud service!"},
                {"role": "assistant", "content": "I can imagine how heartbreaking this must be for you. Wedding photos are irreplaceable memories. Let me immediately escalate this to our data recovery team and personally ensure we explore every possible option to recover your precious photos."}
            ]
        },
        {
            "name": "Mildly Frustrated User", 
            "turns": [
                {"role": "user", "content": "This is a bit annoying. The app keeps logging me out."},
                {"role": "assistant", "content": "I understand that's frustrating. Let me help you fix this login issue right away by checking your account settings."}
            ]
        },
        {
            "name": "Neutral User",
            "turns": [
                {"role": "user", "content": "How do I change my password?"},
                {"role": "assistant", "content": "I'd be happy to help you change your password. You can do this by going to your account settings and clicking on 'Change Password'."}
            ]
        }
    ]
    
    for scenario in scenarios:
        score = await empathy_component.compute_score(scenario["turns"])
        print(f"{scenario['name']}: {score:.3f}")


if __name__ == "__main__":
    print("Enhanced Multi-Objective Reward System - Sentiment Analysis Demo")
    print("================================================================")
    
    # Check if sentiment analysis libraries are available
    try:
        from textblob import TextBlob
        print("✓ TextBlob sentiment analysis available")
    except ImportError:
        print("✗ TextBlob not available - install with: pip install textblob")
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("✓ VADER sentiment analysis available")
    except ImportError:
        print("✗ VADER not available - install with: pip install vaderSentiment")
    
    print("\nRunning tests...")
    asyncio.run(test_sentiment_aware_rewards())
    asyncio.run(test_empathy_component_enhancements())
    
    print("\n" + "="*70)
    print("Test completed! The enhanced reward system provides:")
    print("• Context-aware empathy scoring based on user emotional state")
    print("• Professional tone analysis with sentiment consideration")  
    print("• Emotional intelligence assessment")
    print("• Adaptive responses to different user emotions")
    print("• Improved accuracy through multi-library sentiment analysis")
