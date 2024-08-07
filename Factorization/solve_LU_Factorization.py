from numpy.linalg import inv
from scipy.linalg import lu, lu_solve, lu_factor
from numpy.linalg import det
import numpy as np

## Hàm tạo ma trận hoán vị P từ pivot = linalg.lu_factor(A)[1]
def permutationMatrix(pivot):
    # Khởi tạo P bằng ma trận đơn vị
    P = np.identity(len(pivot))

    # i: index; r: element
    for i, r in enumerate(pivot):
        # pivot[i] = r
        # Hoán đổi dòng r và dòng pivot[i] của ma trận I
        I = np.identity(len(pivot))

        temp    = I[i, :].copy()
        I[i, :] = I[r, :]
        I[r, :] = temp

        P = P.dot(I)
    return P

# Kiểm tra định thức các ma trận con
def check_det(vector_heso):
    if vector_heso[0, 0] == 0:
        return False
    for i in range(2, vector_heso.shape[0] + 1):
        if det(vector_heso[:i, :i]) == 0:
            return False
    return True


# Giải hệ phương trình A*X = B 
def solve_LU_factorization(vector_heso, vector_heso_tudo):
    try:
        # Kiểm tra ma trận trước khi phân rã
        if vector_heso.shape[0] != vector_heso.shape[1]:
            raise ValueError('Ma trận hệ số không phải ma trận vuông')
                
        LU = lu_factor(vector_heso)
        # Ma trận tam giác trên
        U = np.triu(LU[0])
        # Ma trận tam giác dưới
        L = LU[0] - U + np.identity(len(vector_heso_tudo))

        if check_det(vector_heso):
            # Giải A = L*U, ma trận vuông
            x = lu_solve((LU),vector_heso_tudo)
            return x
        else:
            # Giải P.A = L.U, ma trận hình chữ nhật
            # Y = P.T *B *inv(L)
            Y = inv(L) @ vector_heso_tudo @ permutationMatrix(LU[1]).T
            x = Y @ inv(U)
            return x

    except ValueError as ve:
        return (f'ValueError: {ve}')
    except Exception as e:
        return (f'Có lỗi xảy ra: {e}')