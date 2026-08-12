# Code evaluation

`evaluator.py` searches its current directory for `*_results.json` files, generated in stage-3. It then uses gemini-2.5-flash to evaluate the model, ussing parameters specified in the `VertexSettings` dataclass.

Since the logical prompts have objective corrective answers, they are kept in [logic-prompt-keys.json](logic-prompt-keys.json), and are provided to the model during evaluation.