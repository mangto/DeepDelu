import deepdelu, time

network = deepdelu.research.MoreComplexNN([2, 5, 1], deepdelu.tanh)
dataset = deepdelu.datasets.XOR
STEP = 1000
EPOCH = 100000

s = time.time()
for epoch in range(EPOCH):
    c = 0
    for x, y in dataset:
        network.forward(x)
        c += deepdelu.numpy.sum(network.backpropagation(y, lrate=0.1))
    if (epoch%STEP == 0):
        print(f"epoch: {epoch}, cost: {round(c, 6)}, epoch/s: {round(STEP/(time.time()-s),2)}")
        s=time.time()

possible = "0 0 1 1 0"

while __name__ == "__main__":
    try:
        user = input("type two binary > ")
        if (user not in possible or len(user) != 3): continue
        x = [int(user[0]), int(user[2])]
        print(network(x))
    except EOFError:
        break