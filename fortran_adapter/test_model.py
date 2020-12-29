# File: mymodule.py

# def print_args(*args, **kwargs):
#     print("Arguments: ", args)
#     print("Keyword arguments: ", kwargs)

#     return "Returned from mymodule.print_args"

class Model(object):
    def __init__(self, args, kwargs):
        print("Arguments: ", args)
        print("Keyword arguments: ", kwargs)
        print("First Argument: ", args[0])
        self.arg1 = args[0]

    def add_two(self):
        print("Add Two")
        res = str(self.arg1 + 2)
        return res

def make_model(*args, **kwargs):
    # print("Arguments: ", args)
    # print("Keyword arguments: ", kwargs)
    model = Model(args, kwargs)
    result = model.add_two()
    return result


# make_model((12, 'Hi', True), {'message': 'Hello world!'})
