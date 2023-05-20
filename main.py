import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextGenerationPipeline

path = "./models/mosaicml_mpt-7b-instruct"
new_path = "./models"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(new_path, low_cpu_mem_usage=True, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16, max_seq_len=2048)
stop_token_ids = [tokenizer.convert_tokens_to_ids("")]


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(input_ids[0][-1] == stop_id for stop_id in stop_token_ids)


stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device, task='text-generation',
                                       stopping_criteria=stopping_criteria, temperature=0.5, top_p=0.15, top_k=0,
                                       max_length=64, repetition_penalty=1.1)

res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])
