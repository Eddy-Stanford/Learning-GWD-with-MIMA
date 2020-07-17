
def get_batch():
    for i in range(10):
        yield i


batches = get_batch()

def test_generator(batches):
    i = 0
    while True:
        try:
            print(next(batches))
        except StopIteration:
            batches = get_batch()
        i += 1

        if i == 10*5:
            return

test_generator(batches)
