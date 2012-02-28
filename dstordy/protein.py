from itertools import chain
"""
letters are ACDEFGHIKLMNPQRSTVWY
"""
proteinalphabet = "ACDEFGHIKLMNPQRSTVWY"
def get_aa_value(aa):

    index = proteinalphabet.find(aa)
    value = [0]*20

    if index != -1:
        value[index] = 1

    return value

def get_ss_value(ss):
    if ss == "h":
        return [1,0,0]
    elif ss == "e":
        return [0,1,0]
    else:
        return [0,0,1]

def get_ss_index(ss):
    if ss == "h":
        return 0
    elif ss == "e":
        return 1
    else:
        return 2

class Protein:
    def __init__(self):
        self._value = []
        self._s_structure = [] 
    
    def __str__(self):
        return "".join(self._value) + "\n" + "".join(self._s_structure)
        
    def ss_string(self):
        return "".join(self._s_structure)

    def ss_vals(self):
        return [get_ss_index(v) for v in self._s_structure]

    def append_data(self, aa, ss):
        self._value.append(aa)
        self._s_structure.append(ss)
        
    def get_padded(self, padding_size):
        padding = ["-"]*padding_size
        return padding + self._value + padding
    
    def get_encoded_data(self, window):
        padded_protein = self.get_padded(window/2)
        encoded_protein = [get_aa_value(aa) for aa in padded_protein]
        return [ chain(*encoded_protein[i:i+window]) for i in range(len(self._value))]
    
    def get_training_data(self, window):
        windowed_protein = self.get_encoded_data(window)
        encoded_structure = [get_ss_value(ss) for ss in self._s_structure]
        return zip(windowed_protein, encoded_structure)
