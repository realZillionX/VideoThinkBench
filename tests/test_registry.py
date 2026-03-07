import unittest

from data.generate import load_generator_class
from data.registry import TASK_SPECS


class RegistryTests(unittest.TestCase):
    def test_task_registry_uses_new_module_layout(self) -> None:
        self.assertEqual(len(TASK_SPECS), 36)
        for spec in TASK_SPECS.values():
            self.assertTrue(spec.module.startswith("data.visioncentric."))

    def test_all_registered_generators_are_importable(self) -> None:
        for task_name, spec in TASK_SPECS.items():
            generator_class = load_generator_class(spec)
            self.assertIsNotNone(generator_class, msg=task_name)


if __name__ == "__main__":
    unittest.main()
