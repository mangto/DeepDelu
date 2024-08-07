# DeepDelu
학교 발표에 써먹으려고 만든 딥러닝 라이브러리

추가적인 설명이나 자료는 [notion]을 참고해주세요

<div align="center">
<img src="https://github.com/mangto/DeepDelu/blob/main/img/deepdelu.png?raw=true" width="300px" height="300px" title="Github_Logo"/>
<h5>Deep Delu Logo (Generated By Gemini)</h5>
</div>

## Tesnor
``deepdelu.Tensor( data: any )``

numpy에서 사용 가능한 행렬 계산은 대부분 가능함

실제 모델에서 사용하기 위함이라기 보다는, tensor의 개념을 이해하기 위해 만들어짐
 
 
## Research 모델 (오류가 있을 수 있음)
- ``deepdelu.research.LinearEquationNN()``
- ``deepdelu.research.TwoDepthNN( shape:list )``
- ``deepdelu.research.MoreComplexNN( shape:list )``

research 모델들은 모두 ``.forward()``와 ``.backpropagation()``으로 학습가능함
 
 
## 데이터셋
- ``deepdelu.datasets.XOR``
- ``deepdelu.datasets.Reverse``
- ``deepdelu.datasets.Half``
``deepdelu.datasets.unzip()``을 이용하면 X, Y로 분리할 수 있음
 
 
## 활성화 함수
- ``deepdelu.sigmoid( value, Derivative=False )``
- ``deepdelu.relu( value, Derivative=False )``
- ``deepdelu.leaky_relu( value, Derivative=False )``
- ``deepdelu.tanh( value, Derivative=False )``
- ``deepdelu.linear( value, Derivative=False )``
 
 
## Linalg
- ``deepdelu.linalg.norm(array: np.ndarray, p: float = 2.) -> float``
- ``deepdelu.linalg.cosine_similiarity(arr1: np.ndarray, arr2: np.ndarray, p: float = 2.) -> float``

[notion]: https://youthful-kitty-d5e.notion.site/Deep-Delu-Project-155024e556964259b5d5cf5c385dcc93?pvs=4
