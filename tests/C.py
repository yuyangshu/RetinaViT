import A as G

E = G.B()

E.c()

def a():
    print("C")

G.a = a

E.c()
