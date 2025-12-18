import numpy as np


f = lambda x: 8 * np.exp(1 - x) + 7 * np.log(x)

left = 1
right = 2
uncert = 0.23
epsilon = 0.05

F = [1, 1]
n = 0

while F[n+1] < (1 + 2*epsilon) * (right - left) / uncert:
    F.append(F[n+1] + F[n])
    n += 1

N = n
print(f'Number of iterations: {N}')
print('Fibonacci sequence: ')
print(f'{" ".join(map(str, F[:N+2]))}')
print('\n')

lower = 'a'
a = left + (F[N] / F[N+1]) * (right - left)
f_a = f(a)

print('\nleft\tright\ta\tf(a)\tNew Interval\n');
print(f'\t{left:.4f}\t{right:.4f}\t{a:.4f}\t{f_a:.4f}\t{left:.4f},{right:.4f}]\n')

for i in range(1, N+1):
    if i != N:
        rho = 1 - F[N+1-i] / F[N+2-i]
    else:
        rho = 0.5 - epsilon

    if lower == 'a':
        b = a
        f_b = f_a
        a = left + rho * (right - left)
        f_a = f(a)
    else:
        a = b
        f_a = f_b
        b = left + (1 - rho) * (right - left)
        f_b = f(b)

    if f_a < f_b:
        right = b
        lower = 'a'
    else:
        left = a
        lower = 'b'

    print(f'{i}\t{left:.4f}\t{right:.4f}\t{rho:.4f}\t{a:.4f}\t{b:.4f}\t{f_a:.4f}\t{f_b:.4f}\t[{left:.4f}, {right:.4f}]\n')

optimal_x = (left + right) / 2
print(f'\nFinal result: x = {optimal_x:.4f}, {f(optimal_x):4f}\n')
print(f'Final interval: [{left:.4f}, {right:.4f}]\n')
print(f'Interval length: {right-left:.4f}\n')