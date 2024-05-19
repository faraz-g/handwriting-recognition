import torch


class LabelConverter:
    """Convert between text-label and text-index"""

    def __init__(self, character_set: list[str], max_text_length: int):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ["[GO]", "[s]"]  # ['[s]','[UNK]','[PAD]','[GO]']

        self.max_text_length = max_text_length
        self.characters = list_token + character_set
        self.dict = {char: i for i, char in enumerate(self.characters)}

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        batch_max_length = self.max_text_length + 1

        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)

        for i, t in enumerate(text):
            text = list(t)
            text.append("[s]")
            text = [self.dict[char] for char in text]
            batch_text[i][1 : 1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token

        return batch_text, torch.IntTensor(length)

    def decode(self, text_index, length):
        """convert text-index into text-label."""
        texts = []
        for index, _ in enumerate(length):
            text = "".join([self.characters[i] for i in text_index[index, :]])
            texts.append(text)

        return texts
