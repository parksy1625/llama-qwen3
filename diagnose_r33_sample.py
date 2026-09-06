from datasets import load_dataset
from benchmark_llm_standard_v01 import fixed_sample

ds=load_dataset('openai/gsm8k','main',split='test')
xs=fixed_sample(ds,5,20261905)
for i,x in enumerate(xs):
    print(f'=== SAMPLE {i} ===')
    print(x['question'])
    print('--- GOLD TRACE ---')
    print(x['answer'])
