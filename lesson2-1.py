"""linear regression：y = w * x + b"""
import numpy as np
from matplotlib import pyplot as pt


# 损失计算函数
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


# 梯度下降函数
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 对损失函数求偏导
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


# 参数更新函数
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
        print(i)
    return [b, w]

def run():
    points = np.genfromtxt("Label.txt", delimiter=",")
    for i in range(0, len(points)):
        pt.plot(points[i, 0], points[i, 1], '-b', linestyle='', marker='.')
    learning_rate = 0.01
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, w = {1}, error = {2}".format(initial_b, initial_w,
                                                                              compute_error_for_line_given_points(
                                                                                  initial_b, initial_w, points)))
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, w = {2}, error = {3}".format(num_iterations, b, w,
                                                                      compute_error_for_line_given_points(b, w,
                                                                                                          points)))
    X = np.linspace(0, 10, 2, endpoint=True)
    Y = w * X + b
    pt.plot(X, Y, '-r', label='Y=wX+b')
    pt.show()


if __name__ == '__main__':
    run()
