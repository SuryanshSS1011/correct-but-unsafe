### post (13 ops, dataset data_v2)

The plain column flags a fatal memory signal under the interpreter.

| op | known bug | plain status | plain | bounds | tilelens | asan | perturb | poison | any |
|---|---|---|---|---|---|---|---|---|---|
| attention_forward | yes | ok | . | . | . | . | . | . | . |
| chunk_delta_fwd | yes | ok | . | . | . | . | . | . | . |
| chunk_gla_simple | yes | ok | . | . | . | . | . | . | . |
| flash_attn | yes | ok | . | . | . | . | . | . | . |
| mixed_sparse_attention | yes | ok | . | . | . | . | . | . | . |
| parallel_attention | yes | ok | . | . | . | . | . | . | . |
| parallel_retention_attention | yes | ok | . | . | . | . | . | . | . |
| rotary_emb_nopad | yes | ok | . | . | . | . | . | . | . |
| softmax_reducev | yes | ok | . | . | . | . | . | . | . |
| token_attn_mistral | yes | ok | . | . | . | . | . | . | . |
| token_attn_reduceV | yes | ok | . | . | . | . | . | . | . |
| token_softmax_bloom | yes | ok | . | . | . | . | . | . | . |
| token_softmax_llama | yes | ok | . | . | . | . | . | . | . |

Known-bug ops flagged by any detector: 0 of 13.
Known-bug ops flagged by plain: 0 of 13.
Known-bug ops flagged by bounds: 0 of 13.
Known-bug ops flagged by tilelens: 0 of 13.
Known-bug ops flagged by asan: 0 of 13.
Known-bug ops flagged by perturb: 0 of 13.
Known-bug ops flagged by poison: 0 of 13.

### pre (13 ops, dataset data_v2)

The plain column flags a fatal memory signal under the interpreter.

| op | known bug | plain status | plain | bounds | tilelens | asan | perturb | poison | any |
|---|---|---|---|---|---|---|---|---|---|
| attention_forward | yes | exception,ok,rc-6 | FLAG | FLAG | FLAG | FLAG | FLAG | FLAG | FLAG |
| chunk_delta_fwd | yes | rc-6 | FLAG | FLAG | FLAG | FLAG | . | . | FLAG |
| chunk_gla_simple | yes | exception,ok | . | FLAG | FLAG | FLAG | FLAG | . | FLAG |
| flash_attn | yes | exception,ok,rc-6 | FLAG | FLAG | FLAG | FLAG | . | . | FLAG |
| mixed_sparse_attention | yes | exception,ok | . | FLAG | FLAG | FLAG | . | . | FLAG |
| parallel_attention | yes | ok | . | FLAG | FLAG | FLAG | . | . | FLAG |
| parallel_retention_attention | yes | ok | . | FLAG | FLAG | FLAG | . | . | FLAG |
| rotary_emb_nopad | yes | rc-11 | FLAG | FLAG | . | FLAG | . | . | FLAG |
| softmax_reducev | yes | rc-11 | FLAG | FLAG | . | FLAG | . | . | FLAG |
| token_attn_mistral | yes | rc-11 | FLAG | FLAG | . | FLAG | . | . | FLAG |
| token_attn_reduceV | yes | rc-11 | FLAG | FLAG | . | FLAG | . | . | FLAG |
| token_softmax_bloom | yes | rc-6 | FLAG | FLAG | FLAG | FLAG | . | . | FLAG |
| token_softmax_llama | yes | ok,rc-11 | FLAG | FLAG | FLAG | FLAG | FLAG | . | FLAG |

Known-bug ops flagged by any detector: 13 of 13.
Known-bug ops flagged by plain: 9 of 13.
Known-bug ops flagged by bounds: 13 of 13.
Known-bug ops flagged by tilelens: 9 of 13.
Known-bug ops flagged by asan: 13 of 13.
Known-bug ops flagged by perturb: 3 of 13.
Known-bug ops flagged by poison: 1 of 13.

