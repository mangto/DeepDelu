from deepdelu.engine.ahocorasick.trie import trie


class node:
    def __init__(self, key:str) -> None:
        self.key: str = key
        self.next: node = None
        self.end: bool = True
        pass


class ahocorasick:
    def __init__(self) -> None:

        self.node = node('')

        pass

    def add_word(word:str) -> None:

        assert type(word) == str, f"Invalid word type, it must be string (got: {type(word)})"

        previous: node

        for char in word:


            continue


        return