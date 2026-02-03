TASK:
- use the hugging-face-evaluation skill. 
- get the top 50 trending models. 
- review which ones have eval results using the api based on the hugging-face-evaluation skill. 
- skip those that have eval results. 
- extract eval scores from the model card. if available, use them.
- if no eval scores are available in the model card, search the papers linked to the model.
- if no eval scores are available in the papers, search artificial analysis for the model.
- if eval scores are available in artificial analysis, propose PRs to the model repos on Hugging Face. 
OUTPUT FORMAT:
- a correctly formatted table of proposed PRs to model repos on Hugging Face based on the hugging-face-evaluation skill.