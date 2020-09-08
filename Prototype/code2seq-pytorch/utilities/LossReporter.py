
class LossReporter:
    def __init__(self):
        self.epochs = []
        self.current_epoch = 0


    def report(self, loss: int):
        self.epochs[self.current_epoch].append(loss)



