from SM_Runner import make_sm_runner_from_args

if __name__ == '__main__':
    runner = make_sm_runner_from_args()
    _, model = runner.train()
    runner.play(model)
