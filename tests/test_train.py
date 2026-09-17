import unittest

from personaplex_finetuning.train import model_forward_train, write_tensorboard_scalars


class TrainTest(unittest.TestCase):
    def test_uses_lmmodel_forward_train_instead_of_module_forward(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.codes = None

            def forward_train(self, codes):
                self.codes = codes
                return "lm-output"

            def forward(self, _codes):
                raise AssertionError("nn.Module.forward must not be called")

        model = Model()

        self.assertEqual(model_forward_train(model, "codes"), "lm-output")
        self.assertEqual(model.codes, "codes")

    def test_writes_losses_and_training_parameters_to_tensorboard(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.scalars = []

            def add_scalar(self, name, value, step) -> None:
                self.scalars.append((name, value, step))

        writer = Writer()
        write_tensorboard_scalars(
            writer,
            {"step": 2, "loss/total": 1.0, "loss/text": 0.2, "loss/audio_semantic": 0.3,
             "loss/audio_nonsemantic": 0.4, "lr": 2e-5, "grad_norm": 0.5, "gpu_peak_bytes": 1024},
            trainable_parameters=42,
            cpu_threads=1,
        )

        self.assertEqual({name for name, _, _ in writer.scalars}, {
            "loss/total", "loss/text", "loss/audio_semantic", "loss/audio_nonsemantic",
            "train/learning_rate", "train/gradient_norm", "system/gpu_peak_bytes",
            "system/trainable_parameters", "system/cpu_threads",
        })
