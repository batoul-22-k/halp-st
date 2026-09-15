class EarlyStopping:
    def __init__(self, patience: int = 3, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        better = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if better:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

