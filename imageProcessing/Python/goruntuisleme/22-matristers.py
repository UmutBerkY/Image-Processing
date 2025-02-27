import numpy as np

# Bir matris tanımlayalım
A = np.array([[1, 2], [3, 4]])

# Matrisin tersini alalım
A_inv = np.linalg.inv(A)

# Sonucu yazdıralım
print("Matris A:")
print(A)

print("\nMatrisin Tersi A_inv:")
print(A_inv)
