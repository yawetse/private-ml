# Create the directory for prompts if it doesn't exist
import os
os.makedirs('prompts/call_transcripts', exist_ok=True)

# Create random_api_prompt.json
random_prompt = {
    "message_template": [
        {
            "role": "system",
            "content": "You are required to write a realistic call transcript between two people discussing real estate or mortgage-related topics. Ensure the conversation includes typical dialogue elements like greetings, questions, confirmations, and potentially mentions of financial details or personal information in a natural way."
        },
        {
            "role": "user",
            "content": "Generate a call transcript about a real estate showing follow-up or mortgage pre-approval discussion."
        }
    ]
}
with open('prompts/call_transcripts/random_api_prompt.json', 'w') as f:
    import json
    json.dump(random_prompt, f, indent=4)

# Create variation_api_prompt.json
variation_prompt = {
    "message_template": [
        {
            "role": "system",
            "content": "You are a helpful assistant that modifies text while maintaining its core meaning and style. You will be given a call transcript."
        },
        {
            "role": "user",
            "content": "Please rephrase the following call transcript {tone}, keeping it realistic for a real estate or mortgage context:\n{sample}"
        }
    ],
    "replacement_rules": [
        {
            "constraints": {},
            "replacements": {
                "tone": [
                    "in a slightly different way",
                    "using different wording",
                    "with minor variations",
                    "while preserving the key information",
                    "in a similar conversational style"
                ]
            }
        }
    ]
}
# If you want to use blanking/masking similar to the yelp_openai example,
# you would need a more complex prompt structure like the one commented out below.
# variation_prompt = {
#     "message_template": [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant that fills in blanks in text while maintaining the original context and style."
#         },
#         {
#             "role": "user",
#             "content": "Based on the context of a real estate or mortgage call transcript, fill in the blanks in the input sentences. If there are no blanks, output the original input sentences.\nInput: {masked_sample}\nFill-in-Blanks and your answer MUST be exactly {word_count} words:"
#         }
#     ]
# }
with open('prompts/call_transcripts/variation_api_prompt.json', 'w') as f:
    import json
    json.dump(variation_prompt, f, indent=4)

print("Prompt files created in 'prompts/call_transcripts/'")