import unittest
from unittest.mock import patch

import torch

from transformers.common import fill_mask_values, generate_causal_mask
from transformers.multi_head_attention import MultiHeadAttention

BATCH_SIZE = 3
SEQ_LEN = 10
D_MODEL = 32
NHEADS = 4
K_DIM = 16
V_DIM = 16

try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

    _flash_attn_found = True
except ImportError:
    _flash_attn_found = False


class TestMultiHeadAttentionBase(unittest.TestCase):
    def create_mha(
        self,
        key_dim: int | None = None,
        value_dim: int | None = None,
        use_causal_mask: bool = False,
        use_flash_attn: bool = False,
    ):
        return MultiHeadAttention(
            D_MODEL,
            NHEADS,
            dropout=0.0,
            bias=True,
            key_dim=key_dim,
            value_dim=value_dim,
            scale=None,
            use_causal_mask=use_causal_mask,
            use_flash_attn=use_flash_attn,
        ).eval()

    def setUp(self):
        super().setUp()
        self.input = torch.rand((BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.key_input = torch.rand((BATCH_SIZE, SEQ_LEN, K_DIM))
        self.value_input = torch.rand((BATCH_SIZE, SEQ_LEN, V_DIM))

    def assert_cuda_is_available(self):
        self.assertTrue(torch.cuda.is_available())


class TestMultiHeadAttention(TestMultiHeadAttentionBase):
    def test_error_with_no_input(self):
        mha = self.create_mha()
        with self.assertRaises(ValueError):
            mha(None, None, None)

        with self.assertRaises(ValueError):
            mha(self.input, None, None)

        with self.assertRaises(ValueError):
            mha(self.input, self.input, None)

    def test_qkv_packed(self):
        mha = self.create_mha()
        result = mha(self.input, self.input, self.input)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))

    def test_qkv_separate(self):
        mha = self.create_mha(key_dim=K_DIM, value_dim=V_DIM)
        result = mha(self.input, self.key_input, self.value_input)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))

    def test_split_combine_heads(self):
        mha = self.create_mha()
        split_tensor = mha.split_heads(self.input, self.input.size()[0])
        self.assertEqual(
            split_tensor.size(), (BATCH_SIZE, NHEADS, SEQ_LEN, D_MODEL // NHEADS)
        )
        combined_tensor = mha.combine_heads(split_tensor)
        self.assertTrue(torch.all(combined_tensor == self.input))

        split_tensors = mha.split_heads(
            (self.input, self.input, self.input), self.input.size()[0]
        )
        for tensor in split_tensors:
            self.assertEqual(
                tensor.size(), (BATCH_SIZE, NHEADS, SEQ_LEN, D_MODEL // NHEADS)
            )

    def test_qkv_packed_causal_mask(self):
        """Causal mask with qkv packed does not raise error."""
        mha = self.create_mha()
        mask = fill_mask_values(generate_causal_mask(SEQ_LEN)).unsqueeze(0).unsqueeze(0)
        result = mha(self.input, self.input, self.input, mask=mask)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))

    def test_qkv_separate_causal_mask(self):
        """Causal mask with qkv separate does not raise error."""
        mha = self.create_mha(key_dim=K_DIM, value_dim=V_DIM)
        mask = fill_mask_values(generate_causal_mask(SEQ_LEN)).unsqueeze(0).unsqueeze(0)
        result = mha(self.input, self.key_input, self.value_input, mask=mask)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))

    def test_qkv_packed_mask(self):
        """Mask with qkv packed does not raise error."""
        mha = self.create_mha()
        mask = (
            fill_mask_values(
                torch.randint(
                    low=0, high=2, size=(BATCH_SIZE, SEQ_LEN), dtype=torch.float32
                )
            )
            .unsqueeze(1)
            .unsqueeze(-1)
        )
        result = mha(self.input, self.input, self.input, mask=mask)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))

    def test_qkv_separate_mask(self):
        """Mask with qkv separate does not raise error."""
        mha = self.create_mha(key_dim=K_DIM, value_dim=V_DIM)
        mask = (
            fill_mask_values(
                torch.randint(
                    low=0, high=2, size=(BATCH_SIZE, SEQ_LEN), dtype=torch.float32
                )
            )
            .unsqueeze(1)
            .unsqueeze(-1)
        )
        result = mha(self.input, self.key_input, self.value_input, mask=mask)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))


class TestMultiHeadAttentionFlashAttention(TestMultiHeadAttentionBase):
    @patch("transformers.multi_head_attention.flash_attn_qkvpacked_func")
    def test_qkv_packed_flash_attention(self, attn_func_mock):
        self.assert_cuda_is_available()
        self.assertTrue(_flash_attn_found)
        attn_func_mock.side_effect = flash_attn_qkvpacked_func
        mha = self.create_mha(use_flash_attn=True).to(
            device="cuda", dtype=torch.float16
        )
        input = self.input.to(device="cuda", dtype=torch.float16)
        result = mha(input, input, input)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        attn_func_mock.assert_called_once()
        self.assertFalse(attn_func_mock.call_args.kwargs["causal"])

    @patch("transformers.multi_head_attention.flash_attn_qkvpacked_func")
    def test_qkv_packed_flash_attention_causal_mask(self, attn_func_mock):
        self.assert_cuda_is_available()
        self.assertTrue(_flash_attn_found)
        attn_func_mock.side_effect = flash_attn_qkvpacked_func
        mha = self.create_mha(use_flash_attn=True, use_causal_mask=True).to(
            device="cuda", dtype=torch.float16
        )
        input = self.input.to(device="cuda", dtype=torch.float16)
        result = mha(input, input, input)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        attn_func_mock.assert_called_once()
        self.assertTrue(attn_func_mock.call_args.kwargs["causal"])

    @patch("transformers.multi_head_attention.flash_attn_func")
    def test_qkv_separate_flash_attention(self, attn_func_mock):
        self.assert_cuda_is_available()
        self.assertTrue(_flash_attn_found)
        attn_func_mock.side_effect = flash_attn_func
        mha = self.create_mha(use_flash_attn=True, key_dim=K_DIM, value_dim=V_DIM).to(
            device="cuda", dtype=torch.float16
        )
        query = self.input.to(device="cuda", dtype=torch.float16)
        key = self.key_input.to(device="cuda", dtype=torch.float16)
        value = self.value_input.to(device="cuda", dtype=torch.float16)
        result = mha(query, key, value)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        attn_func_mock.assert_called()
        self.assertFalse(attn_func_mock.call_args.kwargs["causal"])

    @patch("transformers.multi_head_attention.flash_attn_func")
    def test_qkv_separate_flash_attention_causal_mask(self, attn_func_mock):
        self.assert_cuda_is_available()
        self.assertTrue(_flash_attn_found)
        attn_func_mock.side_effect = flash_attn_func
        mha = self.create_mha(
            use_flash_attn=True, key_dim=K_DIM, value_dim=V_DIM, use_causal_mask=True
        ).to(device="cuda", dtype=torch.float16)
        query = self.input.to(device="cuda", dtype=torch.float16)
        key = self.key_input.to(device="cuda", dtype=torch.float16)
        value = self.value_input.to(device="cuda", dtype=torch.float16)
        result = mha(query, key, value)
        self.assertEqual(result.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        attn_func_mock.assert_called_once()
        self.assertTrue(attn_func_mock.call_args.kwargs["causal"])

    @patch("transformers.multi_head_attention.flash_attn_qkvpacked_func")
    def test_qkv_packed_mask_before_flash_attention(self, attn_func_mock):
        """Masking is performed before flash attention."""
        self.assert_cuda_is_available()
        self.assertTrue(_flash_attn_found)
        attn_func_mock.side_effect = flash_attn_qkvpacked_func

        mha = self.create_mha(use_flash_attn=True).to(
            device="cuda", dtype=torch.float16
        )
        input = self.input.to(device="cuda", dtype=torch.float16)
        mask = (
            torch.randint(low=0, high=2, size=(BATCH_SIZE, SEQ_LEN))
            .to(device="cuda", dtype=torch.float16)
            .unsqueeze(-1)
        )
        mask = fill_mask_values(mask)

        count = [0]
        original = torch.Tensor.__iadd__

        def mask_before_attention_called(other_tensor):
            attn_func_mock.assert_not_called()
            if torch.equal(other_tensor, mask):
                count[0] += 1
            return original(other_tensor)

        with patch.object(
            torch.Tensor, "__iadd__", side_effect=mask_before_attention_called
        ):
            mha(input, input, input, mask=mask)
            self.assertEqual(count.pop(), 1)

    @patch("transformers.multi_head_attention.flash_attn_func")
    def test_qkv_separate_mask_before_flash_attention(self, attn_func_mock):
        """Masking is performed 3 times before flash attention."""
        self.assert_cuda_is_available()
        self.assertTrue(_flash_attn_found)
        attn_func_mock.side_effect = flash_attn_func

        mha = self.create_mha(use_flash_attn=True, value_dim=V_DIM, key_dim=K_DIM).to(
            device="cuda", dtype=torch.float16
        )
        query = self.input.to(device="cuda", dtype=torch.float16)
        key = self.key_input.to(device="cuda", dtype=torch.float16)
        value = self.value_input.to(device="cuda", dtype=torch.float16)
        mask = (
            torch.randint(low=0, high=2, size=(BATCH_SIZE, SEQ_LEN))
            .to(device="cuda", dtype=torch.float16)
            .unsqueeze(-1)
        )
        mask = fill_mask_values(mask)

        count = [0]
        original = torch.Tensor.__iadd__

        def mask_before_attention_called(other_tensor):
            attn_func_mock.assert_not_called()
            if torch.equal(other_tensor, mask):
                count[0] += 1
            return original(other_tensor)

        with patch("torch.Tensor.__iadd__", side_effect=mask_before_attention_called):
            mha(query, key, value, mask=mask)
            self.assertEqual(count.pop(), 3)

    def test_qkv_packed_flash_attention_masked(self):
        """Masking before flash attention does not raise errors with expected input."""
        self.assert_cuda_is_available()
        self.assertTrue(_flash_attn_found)

        mha = self.create_mha(use_flash_attn=True).to(
            device="cuda", dtype=torch.float16
        )
        input = self.input.to(device="cuda", dtype=torch.float16)
        mask = (
            torch.randint(low=0, high=2, size=(BATCH_SIZE, SEQ_LEN))
            .to(device="cuda", dtype=torch.float16)
            .unsqueeze(-1)
        )
        mask = fill_mask_values(mask)

        result_1 = mha(input, input, input, mask=mask)
        result_2 = mha(input, input, input)
        result_3 = mha(input, input, input)
        self.assertEqual(result_1.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.assertEqual(result_2.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.assertTrue(torch.equal(result_2, result_3))
        self.assertFalse(torch.equal(result_1, result_2))

    def test_qkv_separate_flash_attention_masked(self):
        """Masking before flash attention does not raise errors with expected input."""
        self.assert_cuda_is_available()
        self.assertTrue(_flash_attn_found)
        mha = self.create_mha(use_flash_attn=True, value_dim=V_DIM, key_dim=K_DIM).to(
            device="cuda", dtype=torch.float16
        )
        query = self.input.to(device="cuda", dtype=torch.float16)
        key = self.key_input.to(device="cuda", dtype=torch.float16)
        value = self.value_input.to(device="cuda", dtype=torch.float16)
        mask = (
            torch.randint(low=0, high=2, size=(BATCH_SIZE, SEQ_LEN))
            .to(device="cuda", dtype=torch.float16)
            .unsqueeze(-1)
        )
        mask = fill_mask_values(mask)

        result_1 = mha(query, key, value, mask=mask)
        result_2 = mha(query, key, value)
        result_3 = mha(query, key, value)
        self.assertEqual(result_1.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.assertEqual(result_2.shape, (BATCH_SIZE, SEQ_LEN, D_MODEL))
        self.assertTrue(torch.equal(result_2, result_3))
        self.assertFalse(torch.equal(result_1, result_2))
