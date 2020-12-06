class GetterArgsFunctional:
    """A Mathematical functional with multiple get parameterless functions and multiple constants"""

    def __init__(self, inner_function, getters, *args):
        """
        :param inner_function: The function to apply on the parameters (arguments according to their number and order)
        :param getters: a list or tuple of parameterless functions
        :param args: zero or more constant variables
        """
        self.inner_function = inner_function
        if callable(getters):
            self.getters = [getters]
        else:
            assert isinstance(getters, list) or isinstance(getters, tuple)
            self.getters = getters
        self.args = args
    def __call__(self):
        return self.inner_function(*[g() for g in self.getters] , *self.args)


class AdjustedMultiplier(GetterArgsFunctional):
    def __init__(self, fn, c):
        super(AdjustedMultiplier, self).__init__(lambda x, y: x * y, fn, c)


class AdjustedDivisor(GetterArgsFunctional):
    """ returns c/fn()"""
    def __init__(self, fn, c):
        super(AdjustedDivisor, self).__init__(lambda x, y: y / x, fn, c)


def sum_many(*args):
    """ Enables adding up pytorch tensors with int/float"""
    s = 0
    for a in args:
        s = s + a
    return s


class SumMany(GetterArgsFunctional):
    def __init__(self, funcs, *args):
        super(SumMany, self).__init__(sum_many, funcs, *args)


class SumAdjustedMultipliers(GetterArgsFunctional):
    def __init__(self, f1, f2, c1, c2):
        super(SumAdjustedMultipliers, self).__init__(lambda g1, g2, d1, d2: g1 * d1 + g2 * d2, [f1, f2], c1, c2)


if __name__ == '__main__':
    ws = []
    for i in range(2, 4):
        inner = lambda a,b,c,d: (a,b,c,d)
        o1 = lambda :3
        o2 = lambda :2
        ws.append(GetterArgsFunctional(inner, [o1,o2], i**2, i**3))

    for i in range(5,7):
        o = lambda :7
        ws.append(AdjustedMultiplier(o, i))
        ws.append(AdjustedDivisor(o, i))

    for w in ws:
        print(w())
