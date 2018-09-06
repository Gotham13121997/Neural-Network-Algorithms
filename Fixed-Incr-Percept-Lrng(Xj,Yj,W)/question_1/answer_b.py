"""
Created by Gotham on 06-09-2018.
"""
import answer_a
inputs = [[0.25, 0.353],
          [0.25, 0.471],
          [0.5, 0.353],
          [0.5, 0.647],
          [0.75, 0.705],
          [0.75, 0.882],
          [1, 0.705],
          [1, 1]]

outputs = [0, 1, 0, 1, 0, 1, 0, 1]
fipl = answer_a.FixedIncrPerceptron(inputs=inputs, outputs=outputs, bias=1, learning_rate=0.5)
for i in range(60000):
    fipl.train()
print(fipl.weights)
print(fipl.predict([0.25, 0.353]))
