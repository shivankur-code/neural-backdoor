import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lazarus_reader_lib as L  # noqa: E402


def tiny():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    return LlamaForCausalLM(cfg).eval().requires_grad_(False)


def test_batched_states_match_unbatched():
    model = tiny()
    seqs = [[1, 2, 3], [4, 5, 6, 7, 8], [9]]
    batched = L.gate_states(model, seqs)
    for i, s in enumerate(seqs):
        assert torch.allclose(batched[i], L.gate_states(model, [s])[0], atol=1e-5)
    allb = L.all_states(model, seqs)
    assert allb.shape[0] == 9
    assert torch.allclose(allb[3:8], L.all_states(model, [seqs[1]]), atol=1e-5)


def test_reparameterize_preserves_function_and_reader_scores():
    model = tiny()
    x = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])
    before = model(input_ids=x).logits
    s0, t0 = L.neuron_token_scores(model)
    L.reparameterize(model, [0, 5, 7], c=8.)
    after = model(input_ids=x).logits
    s1, t1 = L.neuron_token_scores(model)
    assert torch.allclose(before, after, atol=1e-4)
    assert torch.allclose(s0, s1, atol=1e-5) and torch.equal(t0, t1)
