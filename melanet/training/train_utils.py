from transformers import TrainerCallback


class TrainerPhaseDetectorCallback(TrainerCallback):
    def __init__(self):
        self.__trainer_phase: str = "train"

    def __phase_from_trainer_control(self, control):
        self._set_trainer_phase("eval" if control.should_evaluate else "train")
        return self.__trainer_phase

    def get_trainer_phase(self) -> str:
        return self.__trainer_phase

    def _set_trainer_phase(self, phase: str) -> None:
        self.__trainer_phase = phase

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