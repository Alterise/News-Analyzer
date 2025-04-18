from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name, device='cpu', use_sbert=False):
        self.model_name = model_name
        self.device = device
        self.use_sbert = use_sbert
        
        if use_sbert:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
        else:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
    
    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        if self.use_sbert:
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            inputs = self.tokenizer(
                texts, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()