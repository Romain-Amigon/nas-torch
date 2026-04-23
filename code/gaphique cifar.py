import matplotlib.pyplot as plt
import warnings

Trans_x = [0, 0, 0, 0, 1, 1, 4, 5, 14]
Trans_y = [46.83, 56.83, 59.55, 59.89, 59.99, 61.87, 61.91, 64.09, 64.09]

Trans_abc_x = [14, 15, 17, 18, 34]
Trans_abc_y = [64.09, 64.61, 65.09, 65.52, 65.52]

abc_x = [2, 3, 4, 9, 12, 15, 29]
abc_y = [56.83, 57.97, 59.24, 60.43, 61.88, 62.79, 62.79]

sa_x = [31, 33, 99]
sa_y = [56.73, 57.21, 57.21]

warnings.warn("Initializing zero-element tensors is a no-op")
print("Début de l'affinage ABC sur proxy CIFAR-10...")
print("Début de la recherche Recuit simulee sur proxy CIFAR-10...")

plt.figure(figsize=(12, 7))

plt.plot(Trans_x, Trans_y, marker='o', linestyle='-', color='blue', drawstyle='steps-post', label='Recherche Transformer (Hybride)')
plt.plot(Trans_abc_x, Trans_abc_y, marker='s', linestyle='-', color='green', drawstyle='steps-post', label='Affinage ABC (Hybride)')

plt.axvline(x=14.5, color='red', linestyle='--', label='Transition Transformer -> ABC')

plt.plot(abc_x, abc_y, marker='^', linestyle='-', color='orange', drawstyle='steps-post', label='Affinage ABC (Seul)')

plt.plot(sa_x, sa_y, marker='d', linestyle='-', color='purple', drawstyle='steps-post', label='Recherche Recuit Simulé')

plt.title('SEED 43')
plt.xlabel('Itérations')
plt.ylabel('Accuracy')

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()