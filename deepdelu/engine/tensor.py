import numpy as np

class Tensor:
    def __init__(self, data:any) -> None:

        if type(data) == Tensor: data = data.data
        self.data = np.array(data)
        self.index = 0

        pass

    def __eq__(self, value: object) -> bool:
        return (type(value) == Tensor and self.data == value.data).all()
    
    def __str__(self) -> str:
        return f"Tensor({str(self.data)})"
    
    def __repr__(self) -> str:
        return f"Tensor({str(self.data)})"
    
    def __sizeof__(self) -> int:
        return self.data.size
    
    def __getattr__(self, name: str) -> np.ndarray:
        return getattr(self.data, name)
    
    def __getitem__(self, item) -> any:
        return self.data[item]
    
    def __iter__(self):
        return self
    
    
    def __add__(self, value):
        return Tensor(self.data + value)

    def __sub__(self, value):
        return Tensor(self.data - value)

    def __mul__(self, value):
        return Tensor(self.data * value)

    def __truediv__(self, value):
        return Tensor(self.data / value)
    
    def __div__(self, value):
        return Tensor(self.data / value)
    
    def __mod__(self, value):
        return Tensor(self.data % value)
    
    def __floordiv__(self, value):
        return Tensor(self.data // value)
    
    def __pow__(self, value):
        return Tensor(self.data ** value)
    
    def __matmul__(self, value):
        return Tensor(self.data @ value)
    
    
    def __radd__(self, value):
        return Tensor(self.data + value)

    def __rsub__(self, value):
        return Tensor(self.data - value)

    def __rmul__(self, value):
        return Tensor(self.data * value)

    def __rtruediv__(self, value):
        return Tensor(value / self.data)
    
    def __rdiv__(self, value):
        return Tensor(value / self.data)
    
    def __mod__(self, value):
        return Tensor(value % self.data)
    
    def __floordiv__(self, value):
        return Tensor(value // self.data)
    
    def __pow__(self, value):
        return Tensor(value ** self.data)
    
    def __rmatmul__(self, value):
        return Tensor(value @ self.data)
    
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self) -> float:
        if (self.index < len(self.data)):
            data = self.data[self.index]
            self.index += 1
            return data
        else:
            raise StopIteration
        
    def __len__(self) -> int:
        return len(self.data)
    
    def append(self, value: float) -> None:
        self.data = np.append(self.data, value)
        return