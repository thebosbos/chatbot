from flask import Flask, request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import transformers



model_name="bosbos/llama3finetune"


# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
 
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},

)

app = Flask(__name__)

@app.route('/get_response')
def chat_function():
    message = request.args.get('message')
    history = request.args.get('history')
    system_prompt = request.args.get('system_prompt')
    max_new_tokens = int(request.args.get('max_new_tokens'))
    temperature = float(request.args.get('temperature'))

    messages = [{"role":"system","content":system_prompt},
                {"role":"user", "content":message}]
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,)
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(
        prompt,
        max_new_tokens = max_new_tokens,
        eos_token_id = terminators,
        do_sample = True,
        temperature = temperature + 0.1,
        top_p = 0.9,)
    return outputs[0]["generated_text"][len(prompt):]

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)
