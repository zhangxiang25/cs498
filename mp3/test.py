import matplotlib
matplotlib.use('Qt5Agg') # Use the new backend
import matplotlib.pyplot as plt

print(f"Using Matplotlib backend: {matplotlib.get_backend()}")

plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
print("Showing plot...")
plt.show()
print("Plot window closed.")