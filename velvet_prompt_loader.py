import json
import os

def load_prompt_deck(path="prompt_deck.json"):
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("\n📚 Velvet Console – Prompt Deck Loaded")
    print(f"Version: {data.get('version')}")
    print("=" * 50)

    for category, cards in data["categories"].items():
        print(f"\n\n=== {category} ===")
        for card in cards:
            print(f"\n🧩 {card['id']} — {card['title']}")
            print(f"   Purpose: {card['purpose']}")
            print(f"   Prompt: {card['prompt']}")
            print(f"   Tip: {card['quick_start']}")
            print("-" * 50)

if __name__ == "__main__":
    load_prompt_deck()
