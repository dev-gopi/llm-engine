from evaluation.tokenizer_quality import benchmark_tokenizer
from evaluation.tokenizer_compat import compatibility_report, assert_compatible
from tokenizer.encoder import Tokenizer

def tok():
    vocab={'<|pad|>':0,'<|unk|>':1,'<|bos|>':2,'<|eos|>':3,'a':4,'b':5}
    return Tokenizer(vocab,special_tokens={'<|pad|>':0,'<|unk|>':1,'<|bos|>':2,'<|eos|>':3})

def test_quality_and_compat():
    t=tok(); q=benchmark_tokenizer(t,{'unicode':['ab'],'json':['ab'],'code':['ab'],'numbers':['ab'],'language':['ab']})
    assert set(q)=={'unicode','json','code','numbers','language'}
    assert assert_compatible(t,{'tokenizer_fingerprint':t.fingerprint})['compatible']
