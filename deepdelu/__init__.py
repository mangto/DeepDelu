from pathlib import Path


from deepdelu import research
from deepdelu.utils import *
from deepdelu.engine.tensor import Tensor

__all__ = [
    'research'
]

######################### Complete Loading #########################

path = Path(__file__).parent.absolute()

with open(str(path)+"\\greeting.txt", "r", encoding="utf8") as file:
    __greeting__ = file.read()

print(__greeting__)