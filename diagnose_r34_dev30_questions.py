from datasets import load_dataset
from benchmark_llm_standard_v01 import fixed_sample

ds=load_dataset('openai/gsm8k','main',split='test')
xs=fixed_sample(ds,30,20260906)
for i,x in enumerate(xs):
    print(f'=== Q{i} ===')
    print(x['question'])
