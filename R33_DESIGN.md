# General-128 V0.9R3.3 design

Typed semantic graph arithmetic router.

The solver may use only quantities and relations grounded in the user question. Gold answers are evaluator-only.

Relation families:
- `rate_product`: amount = count * rate
- `sum_parts`: total = sum(parts)
- `remaining`: remainder = initial - used
- `fraction_of`: part = whole * fraction
- `repeat_scale`: total = episode_amount * frequency * periods
- `bundle_extras`: base + positive excess counts * extra-rate
- `unit_chain`: inventory -> capacity-normalized quantity -> geometric target

All arithmetic is executed by General-128 learned numeric operator families (`g_add/g_sub/g_mul/g_div`). The language cortex may classify a graph family when deterministic cues are ambiguous, but it no longer emits arithmetic programs or selects arbitrary expressions.
