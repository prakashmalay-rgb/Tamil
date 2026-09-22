import asyncio
import sync_kaggle

code = r'''
p = "write email for leave letter for school"
system_prompt = """You are a senior, native Tamil language expert and professional AI assistant.
Always write formal, natural, grammatically correct Tamil responses directly.
When asked to write a letter or email, immediately draft the complete, formal Tamil letter.
Never invent irrelevant stories or repeat words.

Example:
User: write email for leave letter for school
Assistant:
பொருள்: மருத்துவக் காரணங்களுக்காக விடுப்பு விண்ணப்பம்

மதிப்பிற்குரிய வகுப்பு ஆசிரியர் அவர்களுக்கு,

வணக்கம். என் பெயர் செல்வன் கவின், பத்தாம் வகுப்பு 'அ' பிரிவில் பயின்று வருகிறேன். எனக்கு உடல்நலக் குறைவு மற்றும் காய்ச்சல் ஏற்பட்டுள்ளதால், மருத்துவரின் அறிவுரைப்படி இரண்டு நாட்கள் ஓய்வெடுக்க வேண்டியுள்ளது.

எனவே, வரும் 25-09-2026 முதல் 26-09-2026 வரை எனக்கு விடுப்பு அளித்து உதவுமாறு பணிவுடன் கேட்டுக்கொள்கிறேன். பள்ளிக்குத் திரும்பியவுடன் விடுபட்ட பாடங்களை நிறைவு செய்கிறேன்.

நன்றி.

இப்படிக்கு,
தங்கள் உண்மையுள்ள மாணவன்,
கவின் (பத்தாம் வகுப்பு)."""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": p}
]
formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '<think>\n\n</think>\n'
inputs = tokenizer(formatted_input, return_tensors='pt').to(model.device)
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=320,
        do_sample=True,
        temperature=0.2,
        top_p=0.85,
        repetition_penalty=1.1
    )
res = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("FEW_SHOT_RESULT:")
print(res)
'''

async def main():
    out = await sync_kaggle.execute_remote(code)
    print(out)

if __name__ == "__main__":
    asyncio.run(main())
