import pickle

a = pickle.load(open("reward.pkl","rb"))

print(max(a))
