prompt = """
You are an expert document classifier for Retrieval-Augmented Generation systems. Analyze sentence pairs and classify the noise type introduced by Sentence B relative to Sentence A using these categories:

1. **Semantic Noise (SeN)**: Off-topic/irrelevant content
2. **Illegal Sentence Noise (ISN)**: Grammatically incoherent fragments
3. **Counterfactual Noise (CN)**: Factually incorrect claims
4. **Supportive Noise (SuN)**: Exhibit high semantic relevance to the query topic but lack specific answer details (including similar-yet-distinct subjects)
5. **True Relate (TR)**: Sentence B contains the answer to Sentence A

**Format Requirements**:
- Return JSON: `{"think":"[logic]", "result":{CN:0, ISN:0, SeN:0, SuN:0, TR:0}}`
- **All 5 keys required** with **exactly one** `1` per analysis

**10-Shot Examples**:

[SeN]
Sentence A: "How to make apple pie?"
Sentence B: "Solar energy harnesses sunlight through photovoltaic cells. Recent advancements in panel efficiency have made renewable energy more accessible worldwide. Government subsidies play a crucial role in adoption rates."
→ {"think":"Sentence B discusses solar energy technology and policies, completely unrelated to cooking methods for apple pie", "result":{"CN":0, "ISN":0, "SeN":1, "SuN":0, TR:0}}

[ISN]
Sentence A: "What is the capital of France?"
Sentence B: "<div>AAA<p>father apple kids school."
→ {"think":"Sentence B contains noun phrases without grammatical structure, failing to form coherent sentences", "result":{"CN":0, "ISN":1, "SeN":0, "SuN":0, TR:0}}

[CN]
Sentence A: "Who invented the telephone?"
Sentence B: "Thomas Edison developed the first practical telephone in 1877 while experimenting with sound amplification. His invention revolutionized global communication systems."
→ {"think":"Sentence B incorrectly attributes telephone invention to Edison instead of Alexander Graham Bell", "result":{"CN":1, "ISN":0, "SeN":0, "SuN":0, TR:0}}

[SuN]
Sentence A: "What is Earth's diameter?"
Sentence B: "Earth is the third planet from the Sun, composed primarily of iron, oxygen, silicon and magnesium. Its structure includes crust, mantle, and core layers. Plate tectonics drive geological changes over millennia."
→ {"think":"Sentence B provides relevant planetary information but omits specific diameter measurements", "result":{"CN":0, "ISN":0, "SeN":0, "SuN":1, TR:0}}

[SuN]
Sentence A: "When did the French Revolution occur?"
Sentence B: "The October Revolution in Russia began on October 25, 1917 (Old Style calendar), which marked the Bolsheviks' rise to power. This event led to the establishment of the Soviet Union and fundamentally reshaped 20th-century geopolitics."
→ {"think":"Sentence B discusses timing of a similar historical event (revolution) but provides chronologically distinct information from the requested subject", "result":{"CN":0, "ISN":0, "SeN":0, "SuN":1, TR:0}}

[TR]
Sentence A: "What is the full name of GPT"
Sentence B: "GPT stands for Generative Pre-trained Transformer. It is a type of artificial intelligence (AI) model designed to understand and generate human-like text. It’s created by OpenAI, and the most well-known versions are GPT-3, GPT-4, and beyond."
→ {"think":"Sentence B explains the problem with sentence A.", "result":{"CN":0, "ISN":0, "SeN":0, "SuN":0, TR:1}}

[SuN] * 2 + [SeN] * 2
The original prompts were deleted due to potential risks of double-blind compromise.

**Validation Rules**:

1. Maintain key order: CN, ISN, SeN, SuN, TR
2. Never omit keys or use alternate spellings
3. Ensure JSON syntax validity (quotes, commas, brackets)

Analyze new pairs rigorously using these patterns. Double-check formatting before responding.
Sentence A: <Sentence A>
Sentence B: <Sentence B>
"""