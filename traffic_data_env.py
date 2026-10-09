import numpy as np

STEP_COUNT = 200000
SPAWN_THRESHOLD = 0.9
RANDOM_SEED = 1

np.random.seed(RANDOM_SEED)

x1 = np.random.random(STEP_COUNT)
x2 = np.random.random(STEP_COUNT)
x3 = np.random.random(STEP_COUNT)
x4 = np.random.random(STEP_COUNT)

np.save('x1.npy', x1)
np.save('x2.npy', x2)
np.save('x3.npy', x3)
np.save('x4.npy', x4)

print(f'Saved x1.npy shape={x1.shape}')
print(f'Saved x2.npy shape={x2.shape}')
print(f'Saved x3.npy shape={x3.shape}')
print(f'Saved x4.npy shape={x4.shape}')
print(f'Spawn threshold used in simulation: {SPAWN_THRESHOLD}')
print(f'x1 mean={x1.mean():.4f}, spawn rate={(x1 > SPAWN_THRESHOLD).mean():.4f}')
print(f'x2 mean={x2.mean():.4f}, spawn rate={(x2 > SPAWN_THRESHOLD).mean():.4f}')
print(f'x3 mean={x3.mean():.4f}, spawn rate={(x3 > SPAWN_THRESHOLD).mean():.4f}')
print(f'x4 mean={x4.mean():.4f}, spawn rate={(x4 > SPAWN_THRESHOLD).mean():.4f}')
