#!/usr/bin/env python3
"""
CLI Tool for Multilingual Sentiment Analysis

Usage:
    python run.py analyze "Your text here"
    python run.py batch file.txt
    python run.py interactive
    python run.py server
"""

import argparse
import sys
from typing import List

from src.models.sentiment_model import MultilingualSentimentAnalyzer
from src.preprocessing.text_processor import TextProcessor


def print_result(result: dict, show_scores: bool = False):
    """Print analysis result"""
    sentiment_emoji = {
        'positive': '😊',
        'neutral': '😐',
        'negative': '😞'
    }.get(result['sentiment'], '❓')

    print(f"\n{sentiment_emoji} {result['sentiment'].upper()}")
    print(f"   Confidence: {result['confidence']:.3f}")

    if show_scores:
        print("\n   Detailed Scores:")
        for label, score in result['scores'].items():
            bar = "█" * int(score * 30)
            print(f"   {label:10s}: {bar} {score:.3f}")


def analyze_command(args):
    """Analyze single text"""
    analyzer = MultilingualSentimentAnalyzer()
    print("🧠 Loading model...")
    analyzer.load_model()

    processor = TextProcessor()

    text = args.text
    if args.clean:
        text = processor.clean_text(text, lowercase=args.lowercase)

    result = analyzer.analyze(text)

    if args.detect_language:
        lang, conf = processor.detect_language(text)
        print(f"\n🌍 Language: {lang} (confidence: {conf:.2f})")

    print_result(result, show_scores=args.verbose)


def batch_command(args):
    """Analyze texts from file"""
    analyzer = MultilingualSentimentAnalyzer()
    print("🧠 Loading model...")
    analyzer.load_model()

    processor = TextProcessor()

    # Read file
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ Error: File '{args.file}' not found")
        sys.exit(1)

    print(f"\n📄 Processing {len(texts)} texts...")

    results = analyzer.analyze_batch(texts, batch_size=args.batch_size)

    # Summary
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    for r in results:
        sentiment_counts[r['sentiment']] += 1

    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(results)) * 100
        emoji = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}[sentiment]
        bar = "█" * int(percentage / 5)
        print(f"  {emoji} {sentiment.capitalize():10s}: {bar} {count} ({percentage:.1f}%)")

    # Detailed results
    if args.verbose:
        print("\n" + "=" * 60)
        print("📝 Detailed Results")
        print("=" * 60)

        for i, (text, result) in enumerate(zip(texts, results), 1):
            sentiment_emoji = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}[result['sentiment']]
            print(f"\n{i}. {text[:60]}...")
            print(f"   {sentiment_emoji} {result['sentiment'].upper()} ({result['confidence']:.3f})")


def interactive_command(args):
    """Interactive mode"""
    analyzer = MultilingualSentimentAnalyzer()
    print("🧠 Loading model...")
    analyzer.load_model()

    processor = TextProcessor()

    print("\n" + "=" * 60)
    print("🔄 Interactive Mode")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            text = input("📝 Enter text: ")

            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not text.strip():
                continue

            # Analyze
            result = analyzer.analyze(text)
            print_result(result, show_scores=True)

            # Detect language
            if args.detect_language:
                lang, conf = processor.detect_language(text)
                print(f"   🌍 Language: {lang} (confidence: {conf:.2f})")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def server_command(args):
    """Start API server"""
    import uvicorn

    print(f"\n🚀 Starting server on http://{args.host}:{args.port}")
    print("   API documentation: http://{}/docs\n".format(
        f"{args.host}:{args.port}"
    ))

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual Sentiment Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py analyze "I love this product!"
  python run.py analyze "素晴らしい！" --clean
  python run.py batch reviews.txt
  python run.py interactive
  python run.py server
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze single text')
    analyze_parser.add_argument('text', help='Text to analyze')
    analyze_parser.add_argument('--clean', action='store_true', help='Clean text before analysis')
    analyze_parser.add_argument('--lowercase', action='store_true', help='Convert to lowercase')
    analyze_parser.add_argument('--detect-language', action='store_true', help='Detect language')
    analyze_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed scores')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze texts from file')
    batch_parser.add_argument('file', help='File containing texts (one per line)')
    batch_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    batch_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed results')

    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
    interactive_parser.add_argument('--detect-language', action='store_true', help='Detect language')

    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    server_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

    args = parser.parse_args()

    if args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'batch':
        batch_command(args)
    elif args.command == 'interactive':
        interactive_command(args)
    elif args.command == 'server':
        server_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
