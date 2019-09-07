import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

reward = pickle.load(open("reward.pkl","rb"))
y = [i*20000 for i in range(1, len(loss) + 1)]
reward = gaussian_filter1d(reward, sigma = 2)

#reward = pickle.load(open("reward.pkl","rb"))
#print(loss[len(loss)-3])

plt.plot(y,loss)
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.show()

