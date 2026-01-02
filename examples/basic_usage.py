"""
Basic Usage Example for Multilingual Sentiment Analyzer

This example demonstrates how to:
1. Load the sentiment model
2. Analyze single text
3. Analyze batch text
4. Handle multiple languages
"""

import torch
from src.models.sentiment_model import MultilingualSentimentAnalyzer


def example_single_analysis():
    """Example of analyzing single text"""
    print("=" * 60)
    print("Multilingual Sentiment Analyzer - Single Text Analysis")
    print("=" * 60)

    # Initialize analyzer
    analyzer = MultilingualSentimentAnalyzer()

    # Load model
    print("\n🧠 Loading model...")
    analyzer.load_model()

    # Example texts in different languages
    examples = [
        ("English", "I absolutely love this product! It's amazing."),
        ("Japanese", "この製品はとても素晴らしいです。大好きです！"),
        ("Spanish", "¡Me encanta este producto! Es increíble."),
        ("French", "J'aime beaucoup ce produit! C'est formidable."),
        ("German", "Ich liebe dieses Produkt! Es ist toll."),
        ("Chinese", "我非常喜欢这个产品！太棒了！"),
        ("Negative", "This is terrible. I hate it."),
        ("Neutral", "The product arrived yesterday."),
    ]

    print("\n" + "=" * 60)
    print("Analyzing Sentiment")
    print("=" * 60)

    for lang, text in examples:
        print(f"\n📝 {lang}: {text}")
        print("-" * 60)

        try:
            result = analyzer.analyze(text)

            print(f"Sentiment: {result['sentiment'].upper()}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"\nScores:")
            for label, score in result['scores'].items():
                bar = "█" * int(score * 30)
                print(f"  {label:10s}: {bar} {score:.3f}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 60)


def example_batch_analysis():
    """Example of batch analysis"""
    print("\n" + "=" * 60)
    print("Batch Analysis Example")
    print("=" * 60)

    # Initialize analyzer
    analyzer = MultilingualSentimentAnalyzer()
    analyzer.load_model()

    # Batch of texts
    texts = [
        "Great service and fast delivery!",
        "The quality is poor and customer service is terrible.",
        "It's okay, nothing special.",
        "Outstanding experience! Highly recommended!",
        "Not worth the price. Disappointed.",
        "Average product, meets expectations."
    ]

    print(f"\nAnalyzing {len(texts)} texts...")

    results = analyzer.analyze_batch(texts, batch_size=32)

    print("\n" + "-" * 60)
    print(f"{'Text':<40} {'Sentiment':<10} {'Confidence':<10}")
    print("-" * 60)

    for text, result in zip(texts, results):
        sentiment_emoji = {
            'positive': '😊',
            'neutral': '😐',
            'negative': '😞'
        }.get(result['sentiment'], '❓')

        print(f"{text[:40]:<40} {result['sentiment']:<10} {result['confidence']:.3f}")

    # Calculate statistics
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    for r in results:
        sentiment_counts[r['sentiment']] += 1

    print("\n" + "-" * 60)
    print("Summary:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(results)) * 100
        bar = "█" * int(percentage / 5)
        emoji = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}[sentiment]
        print(f"  {emoji} {sentiment.capitalize():<10}: {bar} {count} ({percentage:.1f}%)")


def example_real_time_analysis():
    """Example of real-time sentiment analysis"""
    print("\n" + "=" * 60)
    print("Real-time Analysis (simulated)")
    print("=" * 60)

    analyzer = MultilingualSentimentAnalyzer()
    analyzer.load_model()

    # Simulate social media stream
    social_media_posts = [
        "Just tried the new restaurant downtown. Amazing food! 🍕",
        "Waiting for customer support for 2 hours now... 😤",
        "The weather is nice today.",
        "Best purchase I've made this year! So happy!",
        "Product broke after one day. Waste of money.",
    ]

    print("\n📱 Simulated Social Media Stream:")
    print("-" * 60)

    for i, post in enumerate(social_media_posts, 1):
        result = analyzer.analyze(post)

        emoji = {
            'positive': '😊',
            'neutral': '😐',
            'negative': '😞'
        }[result['sentiment']]

        print(f"\n{i}. {post}")
        print(f"   {emoji} {result['sentiment'].upper()} (confidence: {result['confidence']:.2f})")


def example_custom_threshold():
    """Example with custom confidence threshold"""
    print("\n" + "=" * 60)
    print("Custom Confidence Threshold")
    print("=" * 60)

    analyzer = MultilingualSentimentAnalyzer()
    analyzer.load_model()

    texts = [
        "It's pretty good, I guess.",
        "Maybe I like it, maybe not.",
        "Sort of okay, nothing special."
    ]

    threshold = 0.70  # Only show high-confidence results

    print(f"\nUsing confidence threshold: {threshold}")
    print("-" * 60)

    for text in texts:
        result = analyzer.analyze(text)

        if result['confidence'] >= threshold:
            print(f"\n✅ {text}")
            print(f"   {result['sentiment'].upper()} ({result['confidence']:.3f})")
        else:
            print(f"\n⚠️  {text}")
            print(f"   Low confidence ({result['confidence']:.3f}) - prediction uncertain")


if __name__ == "__main__":
    try:
        # Run examples
        example_single_analysis()
        example_batch_analysis()
        example_real_time_analysis()
        example_custom_threshold()

        print("\n" + "=" * 60)
        print("Examples Complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
