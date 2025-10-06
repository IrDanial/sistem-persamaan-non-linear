import numpy as np

# --- Definisi Fungsi Dasar ---
def f1(x, y):
    """Fungsi pertama f1(x, y) = x^2 + xy - 10"""
    return x**2 + x*y - 10

def f2(x, y):
    """Fungsi kedua f2(x, y) = y + 3xy^2 - 57"""
    return y + 3*x*y**2 - 57

# --- Metode Iterasi Titik Tetap ---
# Fungsi g1A dan g2B berdasarkan NIMx = 1
def g1A(x, y):
    """Fungsi iterasi g1A(x, y) dari f1"""
    return np.sqrt(10 - x*y)

def g2B(x, y):
    """Fungsi iterasi g2B(x, y) dari f2"""
    return 57 / (1 + 3*x*y)

def fixed_point_jacobi(x0, y0, tol, max_iter=100):
    """Metode Iterasi Titik Tetap - Jacobi"""
    x, y = x0, y0
    print("--- Menjalankan Iterasi Titik Tetap (Jacobi) ---")
    for i in range(max_iter):
        x_new = g1A(x, y)
        y_new = g2B(x, y)
        
        # Cek konvergensi
        error = max(abs(x_new - x), abs(y_new - y))
        print(f"Iterasi {i+1}: x = {x_new:.7f}, y = {y_new:.7f}, error = {error:.7f}")
        
        if error < tol:
            print(f"\nSolusi ditemukan setelah {i+1} iterasi.")
            return x_new, y_new, i+1
            
        x, y = x_new, y_new
        
    print("Solusi tidak konvergen dalam batas iterasi maksimum.")
    return None, None, max_iter

def fixed_point_seidel(x0, y0, tol, max_iter=100):
    """Metode Iterasi Titik Tetap - Gauss-Seidel"""
    x, y = x0, y0
    print("\n--- Menjalankan Iterasi Titik Tetap (Gauss-Seidel) ---")
    for i in range(max_iter):
        x_old, y_old = x, y
        
        # Gunakan nilai x yang baru untuk menghitung y
        x = g1A(x, y)
        y = g2B(x, y) # Menggunakan x baru
        
        error = max(abs(x - x_old), abs(y - y_old))
        print(f"Iterasi {i+1}: x = {x:.7f}, y = {y:.7f}, error = {error:.7f}")

        if error < tol:
            print(f"\nSolusi ditemukan setelah {i+1} iterasi.")
            return x, y, i+1
            
    print("Solusi tidak konvergen dalam batas iterasi maksimum.")
    return None, None, max_iter
    
# --- Metode Newton-Raphson ---
# Turunan parsial untuk Jacobian
def df1dx(x, y): return 2*x + y
def df1dy(x, y): return x
def df2dx(x, y): return 3*y**2
def df2dy(x, y): return 1 + 6*x*y

def newton_raphson(x0, y0, tol, max_iter=100):
    """Metode Newton-Raphson untuk sistem persamaan non-linear"""
    x, y = x0, y0
    print("\n--- Menjalankan Metode Newton-Raphson ---")
    for i in range(max_iter):
        # Definisikan vektor F dan matriks Jacobian J
        F = np.array([-f1(x, y), -f2(x, y)])
        J = np.array([
            [df1dx(x, y), df1dy(x, y)],
            [df2dx(x, y), df2dy(x, y)]
        ])
        
        # Selesaikan sistem linear J * delta = F
        try:
            delta = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            print("Matriks Jacobian singular. Metode gagal.")
            return None, None, i+1

        x_new = x + delta[0]
        y_new = y + delta[1]
        
        error = np.sqrt(delta[0]**2 + delta[1]**2)
        print(f"Iterasi {i+1}: x = {x_new:.7f}, y = {y_new:.7f}, error = {error:.7f}")

        if error < tol:
            print(f"\nSolusi ditemukan setelah {i+1} iterasi.")
            return x_new, y_new, i+1
            
        x, y = x_new, y_new
        
    print("Solusi tidak konvergen dalam batas iterasi maksimum.")
    return None, None, max_iter

# --- Metode Secant (Quasi-Newton) ---
def secant(x0, y0, tol, max_iter=100):
    """Metode Secant untuk sistem persamaan non-linear"""
    x, y = x0, y0
    h = 1e-6 # Step kecil untuk aproksimasi turunan
    print("\n--- Menjalankan Metode Secant ---")
    for i in range(max_iter):
        # Aproksimasi Jacobian
        j11 = (f1(x + h, y) - f1(x, y)) / h
        j12 = (f1(x, y + h) - f1(x, y)) / h
        j21 = (f2(x + h, y) - f2(x, y)) / h
        j22 = (f2(x, y + h) - f2(x, y)) / h
        
        F = np.array([-f1(x, y), -f2(x, y)])
        J_approx = np.array([[j11, j12], [j21, j22]])
        
        try:
            delta = np.linalg.solve(J_approx, F)
        except np.linalg.LinAlgError:
            print("Matriks Jacobian singular. Metode gagal.")
            return None, None, i+1
            
        x_new = x + delta[0]
        y_new = y + delta[1]
        
        error = np.sqrt(delta[0]**2 + delta[1]**2)
        print(f"Iterasi {i+1}: x = {x_new:.7f}, y = {y_new:.7f}, error = {error:.7f}")
        
        if error < tol:
            print(f"\nSolusi ditemukan setelah {i+1} iterasi.")
            return x_new, y_new, i+1
        
        x, y = x_new, y_new
        
    print("Solusi tidak konvergen dalam batas iterasi maksimum.")
    return None, None, max_iter


# --- Eksekusi Utama ---
if __name__ == "__main__":
    x_init, y_init = 1.5, 3.5
    tolerance = 1e-6

    # 1. Jalankan Jacobi
    x_jacobi, y_jacobi, iter_jacobi = fixed_point_jacobi(x_init, y_init, tolerance)
    
    # 2. Jalankan Seidel
    x_seidel, y_seidel, iter_seidel = fixed_point_seidel(x_init, y_init, tolerance)
    
    # 3. Jalankan Newton-Raphson
    x_newton, y_newton, iter_newton = newton_raphson(x_init, y_init, tolerance)
    
    # 4. Jalankan Secant
    x_secant, y_secant, iter_secant = secant(x_init, y_init, tolerance)
    
    # --- Rangkuman Hasil ---
    print("\n\n" + "="*40)
    print("           RANGKUMAN HASIL AKHIR")
    print("="*40)
    print(f"{'Metode':<25} {'Solusi (x, y)':<25} {'Iterasi'}")
    print("-"*65)
    if x_jacobi is not None:
        print(f"{'IT Jacobi (g1A, g2B)':<25} ({x_jacobi:.6f}, {y_jacobi:.6f}){'':<7} {iter_jacobi}")
    if x_seidel is not None:
        print(f"{'IT Seidel (g1A, g2B)':<25} ({x_seidel:.6f}, {y_seidel:.6f}){'':<7} {iter_seidel}")
    if x_newton is not None:
        print(f"{'Newton-Raphson':<25} ({x_newton:.6f}, {y_newton:.6f}){'':<7} {iter_newton}")
    if x_secant is not None:
        print(f"{'Secant':<25} ({x_secant:.6f}, {y_secant:.6f}){'':<7} {iter_secant}")
    print("="*40)