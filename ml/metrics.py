def sum_dif(A, F) -> float:
    if sum(F) < 0.0001:
        if sum(A) < 0.0001:
            return 0
        else:
            return 100
    else:
        return 100 * abs((sum(A) - sum(F))) / sum(F)


def MAPE(A, F) -> float:
    result = 0
    for i in range(len(A)):
        if F[i] < 0.0001:
            if A[i] < 0.0001:
                result += 0
            else:
                result += 1
        else:
            result += abs((A[i] - F[i]) / F[i])
    return 100 / len(A) * result
