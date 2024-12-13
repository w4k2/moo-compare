from pymoo.util.display.output import Output
from tqdm import tqdm

class ProgressOutput(Output):
    def __init__(self):
        super().__init__()
        self.bar = tqdm(leave=False)

    def update(self, algorithm):
        self.bar.update(1)

    def finalize(self):
        self.bar.close()