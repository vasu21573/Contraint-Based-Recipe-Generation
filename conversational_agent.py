import torch
import transformers
import os
import json


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from transformers import LlamaTokenizer, LlamaForCausalLM

class LLM:
    id=None

    def __init__(self,model_id):
        self.id=model_id

        self.tokenizer=AutoTokenizer.from_pretrained(
            model_id, 
            token=os.getenv("HUGGINGFACE_API_KEY"),
        )

        self.model=AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            token=os.getenv("HUGGINGFACE_API_KEY"),
            device_map="cuda",
            load_in_4bit=True,
        #     quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )

        # tokenizer.chat_template = "user: {user_content} \n system: {system_content}"

        self.pipeline=transformers.pipeline(
            "text-generation",
            tokenizer=self.tokenizer,
            model=self.model
        )

        self.terminators=[
            self.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def make_thread(self,message,persona):
        return [
            {"role": "system", "content": persona},
            {"role": "user", "content": message},
        ]

    def chat(self,message,persona="",max_len=512,return_thread=False):
        thread=self.make_thread(message,persona)

        outputs=self.pipeline(
            thread,
            max_new_tokens=max_len,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.01,
            top_p=0.1,
        )

        if(return_thread):
            return json.dumps(outputs[0]["generated_text"],indent=4)
        
        return outputs[0]["generated_text"][-1]["content"].replace('\n', ' ').replace('\t', ' ').replace('*', '')
