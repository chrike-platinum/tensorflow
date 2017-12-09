__author__ = 'christiaan'


def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))

print(differences([1,0,1],[1,0,0]))