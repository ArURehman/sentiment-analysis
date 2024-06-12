from transformers import AutoTokenizer

class DataPreprocessing:
    def __init__(self, data: list) -> None:
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  
    
    def _tokenize(self) -> dict:
        return self.tokenizer([self.data], padding=True, truncation=True, return_tensors='tf')    
    
    def _order(self, input_data: dict) -> dict:
        data_values = list(input_data.values())
        return {
            'input_ids': data_values[0],
            'attention_mask': data_values[2],
            'token_type_ids': data_values[1]
        }
    
    def preprocess(self) -> dict:
        tokenized_data = self._tokenize()
        return self._order(tokenized_data)
        