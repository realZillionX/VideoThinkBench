import unittest

from training.vlm.rewards.vlm_rewards import reward_eyeballing, reward_maze, reward_visual_puzzle
from training.vlm.train_grpo import custom_reward_manager


class VLMRewardTests(unittest.TestCase):
    def test_reward_eyeballing_accepts_single_letter(self) -> None:
        self.assertEqual(reward_eyeballing(["<answer>C</answer>"], ["C"]), [1.0])
        self.assertEqual(reward_eyeballing(["<answer>blue</answer>"], ["C"]), [-1.0])

    def test_reward_maze_supports_exact_and_partial_match(self) -> None:
        exact = reward_maze(["<answer>[1, 2, 3]</answer>"], ["[1, 2, 3]"])
        partial = reward_maze(["<answer>[1, 2, 9]</answer>"], ["[1, 2, 3]"])
        self.assertEqual(exact, [1.0])
        self.assertGreaterEqual(partial[0], 0.0)
        self.assertLess(partial[0], 1.0)

    def test_reward_visual_puzzle_matches_normalized_text(self) -> None:
        self.assertEqual(reward_visual_puzzle(["<answer>Blue.</answer>"], ["blue"]), [1.0])
        self.assertEqual(reward_visual_puzzle(["<answer>green</answer>"], ["blue"]), [0.0])
        self.assertEqual(reward_visual_puzzle(["   "], ["blue"]), [-1.0])

    def test_custom_reward_manager_dispatches_by_task_group(self) -> None:
        rewards = custom_reward_manager(
            completions=["<answer>blue</answer>", "<answer>C</answer>", "<answer>[1, 2, 3]</answer>"],
            solution=["blue", "C", "[1, 2, 3]"],
            task_group=["visual_puzzle", "eyeballing", "maze"],
        )
        self.assertEqual(rewards, [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
