from transformers import TrainerCallback


class TrainerPhaseDetectorCallback(TrainerCallback):
    def __init__(self):
        self.__trainer_phase = "train"

    def __phase_from_trainer_control(self, control):
        self.__trainer_phase = "eval" if control.should_evaluate else "train"
        return self.__trainer_phase

    def get_trainer_phase(self):
        return self.__trainer_phase

    def on_step_begin(self, args, state, control, **kwargs):
        self.__phase_from_trainer_control(control)

    def on_step_end(self, args, state, control, **kwargs):
        self.__phase_from_trainer_control(control)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.__phase_from_trainer_control(control)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.__phase_from_trainer_control(control)

    def on_evaluate(self, args, state, control, **kwargs):
        self.__phase_from_trainer_control(control)