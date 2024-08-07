import numpy as np
from numpy.linalg import inv, qr, det
# A = Q x R
# Q: Ma trận trực chuẩn, để ma trận trực chuẩn phải trực giao
# R: Ma trận tam giác trên


# Kiểm tra ma trận Q có trực giao 
#     QxQ.T  = Q.TxQ = I, Q x inv(Q) = I
# <=> inv(Q) x Q x Q.T = I.inv(Q) 
# <=> I x Q.T = I.inv(Q)
# <=> Q.T = inv(Q)
# M trận trực giao khi Q.T = inv(Q)

def is_orthogonal(Q):
    try:
        # Kiểm tra xem kích thước của ma trận Q có phải là vuông không
        if Q.shape[0] != Q.shape[1]:
            raise ValueError('Ma trận Q không phải ma trận vuông')
        else:
            if np.allclose(Q.T, inv(Q)):
                return True
            return False
    except ValueError as ve:
        return (f'ValueError: {ve}')
    except Exception as e:
        return (f'Có lỗi xảy ra: {e}')

def norm_each_cols(Q):
    for i in range(Q.shape[1]):
        if np.linalg.norm(Q) != 1:
            return False
    return True



# A = Q x R
# Q: Ma trận trực chuẩn, để ma trận trực chuẩn phải trực giao
# R: Ma trận tam giác trên
A = np.array([[1, 2], [3, 4]])
# Vector hệ số tự do B
B = np.array([7, 8])
# Giải hệ phương trình A*X = B
""" Phân rã QR: A = Q.R
    Hệ pt tuyến tính: A.X = B
    Ta có: Q.R.X = B
    Đặt R.X = Y
    <=> Q.Y = B
    => Y = B.inv(Q)
    => X = Y.inv(R)
    """
def solve_QR_factorization(vector_heso, vector_heso_tudo):
    try:
        # Kiểm tra ma trận trước khi phân rã
        if vector_heso.shape[0] != vector_heso.shape[1]:
            raise ValueError('Ma trận hệ số không phải ma trận vuông')
        
        Q,R = qr(vector_heso)
        # Kiểm tra ma trận Q và R có khả nghịch không?
        if det(Q)!= 0 and det(R)!= 0:
            Y = vector_heso_tudo @ inv(Q)
            X = np.linalg.solve(R, Y)
            return X
        else:
             raise ValueError('Ma trận Q hoặc R không khả nghịch')

    except ValueError as ve:
        return (f'ValueError: {ve}')
    except Exception as e:
        return (f'Có lỗi xảy ra: {e}')
