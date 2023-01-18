
class Options():
    def __init__(self):
        super().__init__()

        self.Input_Path_Test = './RS300/input/'
        self.Target_Path_Test = ''
        self.Result_Path_Test = './RS300/result/'

        self.MODEL_PATH = './model_best.pth'

        self.Num_Works = 4

        self.CUDA_USE = True