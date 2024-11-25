from flask import Flask, render_template, request, jsonify
import torch
from utils.utils import load_dictionary
from algorithm.FMM_BMM import FMM_BMM
from algorithm.HMM import HMMTokenizer
from algorithm.RNN import RNNSegmentation, Vocabulary
from algorithm.RNN import segment_text as rnn_segment_text
from algorithm.BiLSTM import BiLSTMSegmentation, segment_text as lstm_segment_text
from algorithm.Transformers import ChineseSegmentationTransformer,segment_text as transformer_segment_text
from algorithm.NGram import NgramTokenizer
import jieba
from ltp import LTP
import os
from functools import lru_cache

app = Flask(__name__, template_folder='templates')

# 初始化全局变量和模型
def init_models():
    global dictionary, fmm_bmm, hmm_tokenizer, ngram_tokenizer, rnn_tokenizer, lstm_tokenizer, transformer_tokenizer, ltp
    
    dataset_path = './dataset/icwb2-data'
    training_file = os.path.join(dataset_path, 'gold/msr_training_words.utf8')
    dictionary = load_dictionary(training_file)
    
    model_list_dict = {
        "NGRAM": "./algorithm/model/N-Gram",
        "HMM": "./algorithm/model/HMM",
        "RNN": "./algorithm/model/RNN-based/rnn_segmentation_model.pt",
        "LSTM": "./algorithm/model/BiLSTM-based/bilstm_segmentation_model.pt",
        "Transformer": "./algorithm/model/Transformers-based/chinese_segmentation_transformer.pt"
    }
    
    # 初始化分词器
    fmm_bmm = FMM_BMM(dictionary)
    hmm_tokenizer = HMMTokenizer()
    hmm_tokenizer.load_model(model_list_dict['HMM'])
    ngram_tokenizer = NgramTokenizer()
    ngram_tokenizer.load_model(model_list_dict['NGRAM'])
    
    # 加载深度学习模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    rnn_checkpoint = torch.load(model_list_dict['RNN'], map_location=device)
    rnn_tokenizer = RNNSegmentation(
        rnn_checkpoint['vocab'].get_vocab_size(),
        rnn_checkpoint['vocab'].get_tag_size(),
        embedding_dim=128,
        hidden_dim=256
    ).to(device)
    rnn_tokenizer.load_state_dict(rnn_checkpoint['model_state_dict'])
    rnn_tokenizer.eval()
    
    lstm_checkpoint = torch.load(model_list_dict['LSTM'], map_location=device)
    lstm_tokenizer = BiLSTMSegmentation(
        lstm_checkpoint['vocab'].get_vocab_size(),
        lstm_checkpoint['vocab'].get_tag_size(),
        embedding_dim=256,
        hidden_dim=512
    ).to(device)
    lstm_tokenizer.load_state_dict(lstm_checkpoint['model_state_dict'])
    lstm_tokenizer.eval()
    
    transformer_checkpoint = torch.load(model_list_dict['Transformer'], map_location=device)
    transformer_tokenizer = ChineseSegmentationTransformer(
        vocab_size=transformer_checkpoint['vocab'].get_vocab_size(),
        tag_size=transformer_checkpoint['vocab'].get_tag_size(),
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        dropout=0.1
    ).to(device)
    transformer_tokenizer.load_state_dict(transformer_checkpoint['model_state_dict'])
    transformer_tokenizer.eval()
    
    ltp = LTP()
    
    return {
        'rnn_checkpoint': rnn_checkpoint,
        'lstm_checkpoint': lstm_checkpoint,
        'transformer_checkpoint': transformer_checkpoint
    }

# 使用LRU缓存来存储常见的分词结果
@lru_cache(maxsize=1000)
def get_cached_segmentation(text, method):
    """缓存常用的分词结果"""
    if method == 'fmm':
        return ' '.join(fmm_bmm.forward_max_matching(text))
    elif method == 'bmm':
        return ' '.join(fmm_bmm.backward_max_matching(text))
    elif method == 'hmm':
        return ' '.join(hmm_tokenizer.tokenize(text))
    elif method == 'ngram':
        return ' '.join(ngram_tokenizer.tokenize(text))
    elif method == 'jieba':
        return ' '.join(jieba.cut(text))
    elif method == 'ltp':
        output = ltp.pipeline([text], tasks=['cws'])
        return ' '.join(output.cws[0])
    return None

def get_all_segmentations(text):
    """获取所有分词算法的结果"""
    results = {}
    
    # 检查缓存中是否存在结果
    for method in ['fmm', 'bmm', 'hmm', 'ngram', 'jieba', 'ltp']:
        results[method] = get_cached_segmentation(text, method)
    
    # 深度学习模型（不缓存，因为结果可能会随模型更新而变化）
    with torch.no_grad():
        results['rnn'] = ' '.join(rnn_segment_text(rnn_tokenizer, text, checkpoints['rnn_checkpoint']['vocab']))
        results['lstm'] = ' '.join(lstm_segment_text(lstm_tokenizer, text, checkpoints['lstm_checkpoint']['vocab']))
        results['transformer'] = ' '.join(transformer_segment_text(
            transformer_tokenizer, text, checkpoints['transformer_checkpoint']['vocab']
        ))
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    """AJAX 分词接口"""
    try:
        text = request.json.get('text', '').strip()
        if not text:
            return jsonify({'error': '请输入要分词的文本'}), 400
            
        results = get_all_segmentations(text)
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"分词错误: {str(e)}")
        return jsonify({'error': '分词过程出现错误，请重试'}), 500

if __name__ == '__main__':
    checkpoints = init_models()
    app.run(debug=True, port=4998, threaded=True)