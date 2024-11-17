
class tokenizer():
    def __init__(self,text, desired_vocab_size=156):
        self.desired_vocab_size = desired_vocab_size 
        
        self.text = text
        

        

        


    def base_encode(self,text):
    # ISO-8859-1
        iso_8859_1_chars = (
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            'abcdefghijklmnopqrstuvwxyz'
            'ÀÁÇÈÉÊÔ'
            'àâçèéêôùû'
            " !,.:?''-"
            '\n'
        )
        
        char_to_int = {char: idx for idx, char in enumerate(iso_8859_1_chars)}
        
        encoded_text = [char_to_int[char] for char in text if char in char_to_int]
        
        return encoded_text
    

    def base_decode(self,encoded_text):
    # ISO-8859-1
        iso_8859_1_chars = (
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            'abcdefghijklmnopqrstuvwxyz'
            'ÀÁÇÈÉÊÔ'
            'àâçèéêôùû'
            " !,.:?''-"
            '\n'
        )
        
        int_to_char = {idx: char for idx, char in enumerate(iso_8859_1_chars)}
        
        decoded_text = ''.join([int_to_char[idx] for idx in encoded_text])
        
        return decoded_text
    
    def clean_text(self,text):
    
        allowed_chars = (
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            'abcdefghijklmnopqrstuvwxyz'
            'ÀÁÇÈÉÊÔ'
            'àâçèéêôùû'
            " !,.:?''-"
            '\n'
        )
        
        cleaned_text = ''.join([char if char in allowed_chars else ' ' for char in text])
        
        return cleaned_text




    
    
    def initialisation(self):
        self.text = self.clean_text(self.text)

        self.chars = sorted(list(set(self.text)))
        self.vocab_size=len(self.chars)

        self.num_merged = self.desired_vocab_size - self.vocab_size -1 #number of new tokens we want to add 
        print(f"list of characters being used : {self.chars} lenght of vocab : {self.vocab_size} number of tokens that will be created : {self.num_merged}")
        
        return self.text, self.desired_vocab_size
        
    def encode(self,text,table):
        tokens = self.base_encode(text)
        
        
        new_tokens=[]
    
        k=0
        while k < self.num_merged:
            i = 0
            k+=1
            if k>1:
                tokens=new_tokens
                new_tokens=[]
            while i < len(tokens):
                
                if i<len(tokens)-1 and (tokens[i],tokens[i+1]) in table.keys():
                    
                    new_tokens.append(table[(tokens[i],tokens[i+1])])
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1

        return new_tokens
    
    def decode(self,tokens,table):
        i=0
        while i < self.num_merged:
            i+=1
            if i >1:
                tokens = new_tokens
            new_tokens=[]
            for token in tokens:
                
                if token in table.keys():
                    new_tokens.append(table[token][0])
                    new_tokens.append(table[token][1])
                else:
                    new_tokens.append(token)

        decoded = [self.base_decode([c])[0] for c in new_tokens]

        return ''.join(decoded)
    
    def replace_pair(self,ids, pair, idx):
        new_ids=[]
        i=0
        while i < len(ids):
            if i<len(ids)-1 and (ids[i],ids[i+1])==pair:
                new_ids.append(idx)
                
                i+=2
            else:
                new_ids.append(ids[i])
                i+=1

        return new_ids
    
    def top_pair(self,tokens):
        stats = self.pair(tokens)
        stats = [(value,key) for value,key in zip(stats.values(),stats.keys())]
        sort = sorted(stats, reverse=True)
        top_pair = sort[0]
        return top_pair[1]
    
    def pair(self,tokens):
        pairs = {}
        tokens = [str(token) for token in tokens]
        for i in range(len(tokens)-1):
            text = ' '.join(tokens)
            pairs[(int(tokens[i]),int(tokens[i+1]))]= text.count(tokens[i]+' '+tokens[i+1])
        return pairs
    
    def train_BPE(self, length):
        tokens = self.base_encode(self.text[:length])

        i=0
        decoding_table={}
        encoding_table={}
        while i < self.num_merged:
            i+=1
            idx= self.vocab_size+i
            top = self.top_pair(tokens)
            
            decoding_table[idx]= top
            encoding_table[top]= idx

            tokens = self.replace_pair(tokens,top,idx)

        return decoding_table, encoding_table