import json
import sys
import os

def extract_tokenizer(tokenizer_json_path, tokenizer_config_path, output_dir):
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    model = data.get('model', {})
    vocab = model.get('vocab', {})
    merges = model.get('merges', [])
    
    vocab_path = os.path.join(output_dir, 'vocab.txt')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f'{token}\n')
    
    merges_path = os.path.join(output_dir, 'merges.txt')
    with open(merges_path, 'w', encoding='utf-8') as f:
        for merge in merges:
            if isinstance(merge, list) and len(merge) == 2:
                f.write(f'{merge[0]} {merge[1]}\n')
    
    print(f'Extracted vocab ({len(vocab)} tokens) to {vocab_path}')
    print(f'Extracted merges ({len(merges)} pairs) to {merges_path}')
    
    special_tokens = {}
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        for key in ['bos_token', 'eos_token', 'unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token']:
            if key in config and config[key]:
                special_tokens[key] = config[key]
        
        print(f'Special tokens: {special_tokens}')
    
    model_json = {
        "model_type": "embedding",
        "params": {
            "encoder_param": "jina_v5_nano_text_matching.ncnn.param",
            "encoder_bin": "jina_v5_nano_text_matching.ncnn.bin"
        },
        "tokenizer": {
            "type": "bpe",
            "vocab_file": "vocab.txt",
            "merges_file": "merges.txt",
            "bos": special_tokens.get('bos_token', ''),
            "eos": special_tokens.get('eos_token', ''),
            "pad": special_tokens.get('pad_token', ''),
            "unk": special_tokens.get('unk_token', ''),
            "additional_special_tokens": []
        },
        "setting": {
            "embed_dim": 768,
            "rope_head_dim": 64,
            "rope_theta": 100000.0
        }
    }
    
    model_json_path = os.path.join(output_dir, 'model.json')
    with open(model_json_path, 'w', encoding='utf-8') as f:
        json.dump(model_json, f, indent=4)
    
    print(f'Generated model.json at {model_json_path}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python extract_tokenizer.py <tokenizer.json> <output_dir> [tokenizer_config.json]')
        sys.exit(1)
    
    tokenizer_json = sys.argv[1]
    output_dir = sys.argv[2]
    tokenizer_config = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(tokenizer_json), 'tokenizer_config.json')
    
    extract_tokenizer(tokenizer_json, tokenizer_config, output_dir)