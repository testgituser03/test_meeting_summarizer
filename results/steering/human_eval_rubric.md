# Human Evaluation Rubric — Action Item Clarity (1-5)

Score each generated summary for whether concrete next actions are clear.

- 1: No actionable item; vague or purely descriptive.
- 2: Weakly actionable; actor/action unclear.
- 3: Some action present, but missing owner, timing, or specificity.
- 4: Clear action item(s) with mostly clear owner and intent.
- 5: Highly clear, specific action item(s) with explicit intent/owner and minimal ambiguity.

Protocol:
1. Use only the first 50 samples from the template.
2. Rate every (layer, alpha, sample) row.
3. Keep the rubric constant across all settings.
4. Save completed CSV and pass via --ratings_csv.
