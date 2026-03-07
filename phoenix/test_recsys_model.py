# Copyright 2026 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grok import TransformerConfig, make_recsys_attn_mask
from recsys_model import HashConfig, PhoenixModelConfig, RecsysModelOutput
from runners import ModelRunner, RecsysInferenceRunner, create_example_batch


class TestMakeRecsysAttnMask:
    """Tests for the make_recsys_attn_mask function."""

    def test_output_shape(self):
        """Test that the output has the correct shape [1, 1, seq_len, seq_len]."""
        seq_len = 10
        candidate_start_offset = 5

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)

        assert mask.shape == (1, 1, seq_len, seq_len)

    def test_user_history_has_causal_attention(self):
        """Test that user+history positions (before candidate_start_offset) have causal attention."""
        seq_len = 8
        candidate_start_offset = 5

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        for i in range(candidate_start_offset):
            for j in range(candidate_start_offset):
                if j <= i:
                    assert mask_2d[i, j] == 1, f"Position {i} should attend to position {j}"
                else:
                    assert mask_2d[i, j] == 0, (
                        f"Position {i} should NOT attend to future position {j}"
                    )

    def test_candidates_attend_to_user_history(self):
        """Test that candidates can attend to all user+history positions."""
        seq_len = 8
        candidate_start_offset = 5

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        for candidate_pos in range(candidate_start_offset, seq_len):
            for history_pos in range(candidate_start_offset):
                assert mask_2d[candidate_pos, history_pos] == 1, (
                    f"Candidate at {candidate_pos} should attend to user+history at {history_pos}"
                )

    def test_candidates_attend_to_themselves(self):
        """Test that candidates can attend to themselves (self-attention)."""
        seq_len = 8
        candidate_start_offset = 5

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        for candidate_pos in range(candidate_start_offset, seq_len):
            assert mask_2d[candidate_pos, candidate_pos] == 1, (
                f"Candidate at {candidate_pos} should attend to itself"
            )

    def test_candidates_do_not_attend_to_other_candidates(self):
        """Test that candidates cannot attend to other candidates."""
        seq_len = 8
        candidate_start_offset = 5

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        for query_pos in range(candidate_start_offset, seq_len):
            for key_pos in range(candidate_start_offset, seq_len):
                if query_pos != key_pos:
                    assert mask_2d[query_pos, key_pos] == 0, (
                        f"Candidate at {query_pos} should NOT attend to candidate at {key_pos}"
                    )

    def test_full_mask_structure(self):
        """Test the complete mask structure with a small example."""
        # Sequence: [user, h1, h2, c1, c2, c3]
        # Positions:  0     1   2   3   4   5

        seq_len = 6
        candidate_start_offset = 3

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        # Expected mask structure:
        # Query positions are rows, key positions are columns
        # 1 = can attend, 0 = cannot attend
        #
        #        Keys:  u   h1  h2  c1  c2  c3
        # Query u   :   1   0   0   0   0   0
        # Query h1  :   1   1   0   0   0   0
        # Query h2  :   1   1   1   0   0   0
        # Query c1  :   1   1   1   1   0   0   <- c1 attends to user+history + self
        # Query c2  :   1   1   1   0   1   0   <- c2 attends to user+history + self
        # Query c3  :   1   1   1   0   0   1   <- c3 attends to user+history + self

        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # user
                [1, 1, 0, 0, 0, 0],  # h1
                [1, 1, 1, 0, 0, 0],  # h2
                [1, 1, 1, 1, 0, 0],  # c1: user+history + self
                [1, 1, 1, 0, 1, 0],  # c2: user+history + self
                [1, 1, 1, 0, 0, 1],  # c3: user+history + self
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(
            np.array(mask_2d),
            expected,
            err_msg="Full mask structure does not match expected pattern",
        )

    def test_dtype_preserved(self):
        """Test that the specified dtype is used."""
        seq_len = 5
        candidate_start_offset = 3

        mask_f32 = make_recsys_attn_mask(seq_len, candidate_start_offset, dtype=jnp.float32)
        mask_f16 = make_recsys_attn_mask(seq_len, candidate_start_offset, dtype=jnp.float16)

        assert mask_f32.dtype == jnp.float32
        assert mask_f16.dtype == jnp.float16

    def test_single_candidate(self):
        """Test edge case with a single candidate."""
        seq_len = 4
        candidate_start_offset = 3

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        expected = np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(np.array(mask_2d), expected)

    def test_all_candidates(self):
        """Test edge case where all positions except first are candidates."""
        seq_len = 4
        candidate_start_offset = 1

        mask = make_recsys_attn_mask(seq_len, candidate_start_offset)
        mask_2d = mask[0, 0]

        expected = np.array(
            [
                [1, 0, 0, 0],  # user
                [1, 1, 0, 0],  # c1: user + self
                [1, 0, 1, 0],  # c2: user + self
                [1, 0, 0, 1],  # c3: user + self
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(np.array(mask_2d), expected)


class TestPhoenixModel(unittest.TestCase):
    """Tests for the PhoenixModel ranking model."""

    def setUp(self):
        """Set up test fixtures."""
        self.emb_size = 64
        self.num_actions = 19
        self.history_seq_len = 16
        self.candidate_seq_len = 8
        self.batch_size = 2

        self.hash_config = HashConfig(
            num_user_hashes=2,
            num_item_hashes=2,
            num_author_hashes=2,
        )

        self.config = PhoenixModelConfig(
            emb_size=self.emb_size,
            num_actions=self.num_actions,
            history_seq_len=self.history_seq_len,
            candidate_seq_len=self.candidate_seq_len,
            hash_config=self.hash_config,
            product_surface_vocab_size=16,
            model=TransformerConfig(
                emb_size=self.emb_size,
                widening_factor=2,
                key_size=32,
                num_q_heads=2,
                num_kv_heads=2,
                num_layers=1,
                attn_output_multiplier=0.125,
            ),
        )

    def _create_test_batch(self):
        """Create test batch and embeddings."""
        return create_example_batch(
            batch_size=self.batch_size,
            emb_size=self.emb_size,
            history_len=self.history_seq_len,
            num_candidates=self.candidate_seq_len,
            num_actions=self.num_actions,
            num_user_hashes=self.hash_config.num_user_hashes,
            num_item_hashes=self.hash_config.num_item_hashes,
            num_author_hashes=self.hash_config.num_author_hashes,
            product_surface_vocab_size=self.config.product_surface_vocab_size,
        )

    def test_model_forward_output_shape(self):
        """Test that PhoenixModel forward pass produces correct output shape."""

        def forward(batch, embeddings):
            model = self.config.make()
            return model(batch, embeddings)

        forward_fn = hk.without_apply_rng(hk.transform(forward))

        batch, embeddings = self._create_test_batch()

        rng = jax.random.PRNGKey(0)
        params = forward_fn.init(rng, batch, embeddings)
        output = forward_fn.apply(params, batch, embeddings)

        self.assertIsInstance(output, RecsysModelOutput)
        self.assertEqual(
            output.logits.shape,
            (self.batch_size, self.candidate_seq_len, self.num_actions),
        )

    def test_model_logits_finite(self):
        """Test that model logits are finite (no NaN or Inf)."""

        def forward(batch, embeddings):
            model = self.config.make()
            return model(batch, embeddings)

        forward_fn = hk.without_apply_rng(hk.transform(forward))

        batch, embeddings = self._create_test_batch()

        rng = jax.random.PRNGKey(0)
        params = forward_fn.init(rng, batch, embeddings)
        output = forward_fn.apply(params, batch, embeddings)

        self.assertTrue(jnp.all(jnp.isfinite(output.logits)))

    def test_candidates_scored_independently(self):
        """Test that each candidate is scored independently (candidate isolation).

        A key property of the ranking model is that the score for one candidate
        should not change when another candidate is added or removed.
        """

        def forward(batch, embeddings):
            model = self.config.make()
            return model(batch, embeddings)

        forward_fn = hk.without_apply_rng(hk.transform(forward))

        batch, embeddings = self._create_test_batch()

        rng = jax.random.PRNGKey(42)
        params = forward_fn.init(rng, batch, embeddings)
        output_full = forward_fn.apply(params, batch, embeddings)

        # The same params applied to a batch with only 1 candidate should yield
        # the same score for that candidate as in the full batch
        from recsys_model import RecsysBatch, RecsysEmbeddings

        single_batch = RecsysBatch(
            user_hashes=batch.user_hashes,
            history_post_hashes=batch.history_post_hashes,
            history_author_hashes=batch.history_author_hashes,
            history_actions=batch.history_actions,
            history_product_surface=batch.history_product_surface,
            candidate_post_hashes=batch.candidate_post_hashes[:, :1, :],
            candidate_author_hashes=batch.candidate_author_hashes[:, :1, :],
            candidate_product_surface=batch.candidate_product_surface[:, :1],
        )
        single_embeddings = RecsysEmbeddings(
            user_embeddings=embeddings.user_embeddings,
            history_post_embeddings=embeddings.history_post_embeddings,
            candidate_post_embeddings=embeddings.candidate_post_embeddings[:, :1, :, :],
            history_author_embeddings=embeddings.history_author_embeddings,
            candidate_author_embeddings=embeddings.candidate_author_embeddings[:, :1, :, :],
        )

        output_single = forward_fn.apply(params, single_batch, single_embeddings)

        # Score for candidate 0 should be the same whether or not other candidates are present
        np.testing.assert_array_almost_equal(
            np.array(output_full.logits[:, 0, :]),
            np.array(output_single.logits[:, 0, :]),
            decimal=4,
            err_msg="Candidate scores should be independent of other candidates in the batch",
        )


class TestRecsysInferenceRunner(unittest.TestCase):
    """Tests for the RecsysInferenceRunner (ranking inference runner)."""

    def setUp(self):
        """Set up test fixtures."""
        self.emb_size = 64
        self.num_actions = 19
        self.history_seq_len = 16
        self.candidate_seq_len = 8
        self.batch_size = 2

        self.hash_config = HashConfig(
            num_user_hashes=2,
            num_item_hashes=2,
            num_author_hashes=2,
        )

        self.config = PhoenixModelConfig(
            emb_size=self.emb_size,
            num_actions=self.num_actions,
            history_seq_len=self.history_seq_len,
            candidate_seq_len=self.candidate_seq_len,
            hash_config=self.hash_config,
            product_surface_vocab_size=16,
            model=TransformerConfig(
                emb_size=self.emb_size,
                widening_factor=2,
                key_size=32,
                num_q_heads=2,
                num_kv_heads=2,
                num_layers=1,
                attn_output_multiplier=0.125,
            ),
        )

    def test_runner_initialization(self):
        """Test that the inference runner initializes correctly."""
        runner = RecsysInferenceRunner(
            runner=ModelRunner(
                model=self.config,
                bs_per_device=0.125,
            ),
            name="test_ranking",
        )

        runner.initialize()

        self.assertIsNotNone(runner.params)

    def test_runner_rank_output_shape(self):
        """Test that the ranking runner produces correct output shapes."""
        runner = RecsysInferenceRunner(
            runner=ModelRunner(
                model=self.config,
                bs_per_device=0.125,
            ),
            name="test_ranking",
        )
        runner.initialize()

        batch, embeddings = create_example_batch(
            batch_size=self.batch_size,
            emb_size=self.emb_size,
            history_len=self.history_seq_len,
            num_candidates=self.candidate_seq_len,
            num_actions=self.num_actions,
            num_user_hashes=self.hash_config.num_user_hashes,
            num_item_hashes=self.hash_config.num_item_hashes,
            num_author_hashes=self.hash_config.num_author_hashes,
            product_surface_vocab_size=self.config.product_surface_vocab_size,
        )

        output = runner.rank(batch, embeddings)

        self.assertEqual(
            output.scores.shape,
            (self.batch_size, self.candidate_seq_len, self.num_actions),
        )
        self.assertEqual(output.ranked_indices.shape, (self.batch_size, self.candidate_seq_len))

    def test_runner_ranked_indices_valid(self):
        """Test that ranked_indices are valid permutation indices."""
        runner = RecsysInferenceRunner(
            runner=ModelRunner(
                model=self.config,
                bs_per_device=0.125,
            ),
            name="test_ranking",
        )
        runner.initialize()

        batch, embeddings = create_example_batch(
            batch_size=self.batch_size,
            emb_size=self.emb_size,
            history_len=self.history_seq_len,
            num_candidates=self.candidate_seq_len,
            num_actions=self.num_actions,
            num_user_hashes=self.hash_config.num_user_hashes,
            num_item_hashes=self.hash_config.num_item_hashes,
            num_author_hashes=self.hash_config.num_author_hashes,
            product_surface_vocab_size=self.config.product_surface_vocab_size,
        )

        output = runner.rank(batch, embeddings)

        indices = np.array(output.ranked_indices)
        for b in range(self.batch_size):
            self.assertEqual(sorted(indices[b]), list(range(self.candidate_seq_len)))

    def test_runner_scores_are_probabilities(self):
        """Test that scores are probabilities in [0, 1] after sigmoid."""
        runner = RecsysInferenceRunner(
            runner=ModelRunner(
                model=self.config,
                bs_per_device=0.125,
            ),
            name="test_ranking",
        )
        runner.initialize()

        batch, embeddings = create_example_batch(
            batch_size=self.batch_size,
            emb_size=self.emb_size,
            history_len=self.history_seq_len,
            num_candidates=self.candidate_seq_len,
            num_actions=self.num_actions,
            num_user_hashes=self.hash_config.num_user_hashes,
            num_item_hashes=self.hash_config.num_item_hashes,
            num_author_hashes=self.hash_config.num_author_hashes,
            product_surface_vocab_size=self.config.product_surface_vocab_size,
        )

        output = runner.rank(batch, embeddings)

        scores = np.array(output.scores)
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
