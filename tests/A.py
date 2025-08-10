def a():
    print("A")

def b():
    print("B")

class B:
    def __init__(self):
        self.bla = "A"

    def c(self):
        a()

D = B()

D.c()

# a = b

D.c()
