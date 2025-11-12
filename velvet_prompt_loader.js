const fs = require("fs");

function loadPromptDeck(path = "./prompt_deck.json") {
    if (!fs.existsSync(path)) {
        console.error(`❌ File not found: ${path}`);
        return;
    }

    const raw = fs.readFileSync(path, "utf8");
    const data = JSON.parse(raw);

    console.log("\n📚 Velvet Console – Prompt Deck Loaded");
    console.log(`Version: ${data.version}`);
    console.log("=".repeat(50));

    Object.entries(data.categories).forEach(([category, cards]) => {
        console.log(`\n\n=== ${category} ===`);
        cards.forEach(card => {
            console.log(`\n🧩 ${card.id} — ${card.title}`);
            console.log(`   Purpose: ${card.purpose}`);
            console.log(`   Prompt: ${card.prompt}`);
            console.log(`   Tip: ${card.quick_start}`);
            console.log("-".repeat(50));
        });
    });
}

loadPromptDeck();
