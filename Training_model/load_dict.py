from sklearn.model_selection import train_test_split

def load_cmu_dict(file_path):
    words = []
    phonemes = []
    with open(file_path, 'r') as file:
        for line in file:

            parts = line.strip().split('\t')
            parts = line.strip().split()
            if len(parts) > 1:
                word = parts[0]
                phoneme_seq = parts[1:]
                words.append(word)
                phonemes.append(phoneme_seq)
    return words, phonemes


def split_data(words, phonemes):
    return train_test_split(words, phonemes, train_size=0.8, test_size=0.2, random_state=42)
