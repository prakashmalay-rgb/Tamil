import asyncio
import sync_kaggle

code = """
import torch
model.eval()
model.set_adapter('mastery')
prompt = 'write email for leave letter for school'
msgs = [
    {
        'role': 'system', 
        'content': (
            'You are a senior, native Tamil language expert and professional AI assistant. '
            'Always respond in fluent, grammatically accurate, pure Tamil (தமிழ்). '
            'When asked to write a letter, email, or official document, IMMEDIATELY draft the full, formal letter directly in proper Tamil (பொருள், மதிப்பிற்குரிய ஐயா, முழுமையான கடித உள்ளடக்கம், இப்படிக்கு). '
            'CRITICAL: NEVER output an empty list of bracket placeholders like [நீங்கள் பெயர்] or [உங்கள் முகவரி]. Always write the complete, ready-to-use, professional letter in full.'
        )
    },
    {'role': 'user', 'content': prompt}
]
txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + '<think>' + chr(10) + chr(10) + '</think>' + chr(10)
inputs = tokenizer(txt, return_tensors='pt').to(model.device)
with torch.inference_mode():
    out = model.generate(
        **inputs, 
        max_new_tokens=220, 
        do_sample=True, 
        temperature=0.2, 
        top_p=0.85, 
        repetition_penalty=1.1, 
        pad_token_id=tokenizer.eos_token_id
    )
res = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print('>>> GENERATION_RESULT START <<<')
print(res)
print('>>> GENERATION_RESULT END <<<')
"""

async def run():
    out = await sync_kaggle.execute_remote(code)
    print(out)

if __name__ == "__main__":
    asyncio.run(run())
