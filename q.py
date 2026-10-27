import numpy as np

maze = [
    "S.#.....",
    ".##.###.",
    "....#...",
    ".####.#.",
    "......#.",
    ".####.#G",
    "....#...",
    "###...#."
]

Q = np.zeros((8, 8, 4))
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

alpha = 0.1
gamma = 0.9
epsilon = 0.2

start = (0, 0)
goal = (5, 7)

# Training
for episode in range(3000):
    r, c = start

    for step in range(200):
        # Choose an action
        if np.random.random() < epsilon:
            a = np.random.randint(4)
        else:
            a = np.argmax(Q[r, c])

        nr = r + actions[a][0]
        nc = c + actions[a][1]

        # Check wall or boundary
        if (nr < 0 or nr >= 8 or nc < 0 or nc >= 8
                or maze[nr][nc] == '#'):
            reward = -5
            nr, nc = r, c
        elif (nr, nc) == goal:
            reward = 100
        else:
            reward = -1

        # Q-learning update
        Q[r, c, a] += alpha * (
            reward + gamma * (0 if (nr, nc) == goal
                              else np.max(Q[nr, nc]))
            - Q[r, c, a]
        )

        r, c = nr, nc

        if (r, c) == goal:
            break

# Display best action in every cell
arrows = ['U', 'D', 'L', 'R']

for r in range(8):
    for c in range(8):
        if maze[r][c] == '#':
            print('#', end=' ')
        elif (r, c) == start:
            print('S', end=' ')
        elif (r, c) == goal:
            print('G', end=' ')
        else:
            print(arrows[np.argmax(Q[r, c])], end=' ')
    print()
import matplotlib.pyplot as plt
import numpy as np

# Find the learned path
r, c = start
path = [(r, c)]

for _ in range(100):
    a = np.argmax(Q[r, c])
    r += actions[a][0]
    c += actions[a][1]

    # Stop if the move hits a wall or boundary
    if (r < 0 or r >= 8 or c < 0 or c >= 8
            or maze[r][c] == '#'):
        break

    path.append((r, c))

    if (r, c) == goal:
        break

# Convert maze into numbers for plotting
grid = np.zeros((8, 8))

for r in range(8):
    for c in range(8):
        if maze[r][c] == '#':
            grid[r, c] = 1

# Plot before and after
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

for i in range(2):
    ax[i].imshow(grid, cmap='Greys', vmin=0, vmax=1)
    ax[i].set_xticks(range(8))
    ax[i].set_yticks(range(8))
    ax[i].set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax[i].set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax[i].grid(which='minor', color='gray')
    ax[i].set_xlim(-0.5, 7.5)
    ax[i].set_ylim(7.5, -0.5)

    ax[i].text(0, 0, 'S', ha='center', va='center',
               color='green', fontsize=14, fontweight='bold')
    ax[i].text(7, 5, 'G', ha='center', va='center',
               color='red', fontsize=14, fontweight='bold')

ax[0].set_title("Before Learning")

# Draw learned path
ys, xs = zip(*path)
ax[1].plot(xs, ys, 'o-', color='orange', linewidth=2)

ax[1].set_title("After Q-Learning")

plt.tight_layout()
plt.show()
