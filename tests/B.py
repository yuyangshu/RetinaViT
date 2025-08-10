import A

E = A.B()

E.c()

def a():
    print("C")

A.a = a

E.c()
