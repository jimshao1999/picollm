import copy

import tiktoken

SPECIAL_TOKENS = {
    "<|user_start|>": 50257,
    "<|user_end|>": 50258,
    "<|assistant_start|>": 50259,
    "<|assistant_end|>": 50260,
}


class ChatTokenizer:
    def __init__(self):
        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self) -> tiktoken.Encoding:
        base = tiktoken.encoding_for_model("gpt2")
        return tiktoken.Encoding(
            name="gpt2_chat",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={
                **base._special_tokens,
                **SPECIAL_TOKENS,
            },
        )

    def encode_special(self, token):
        return self.tokenizer.encode_single_token(token)

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a Chat conversation.
        Returns:
        - ids: a list of token ids of this rendered conversation
        - mask: mask = 1 for tokens that the Assistant is expected to train on.
        """
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # edge case: the first message is a system prompt
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)  # avoid mutating the original
            messages = conversation["messages"]
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        user_start, user_end = self.encode_special(
            "<|user_start|>"
        ), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special(
            "<|assistant_start|>"
        ), self.encode_special("<|assistant_end|>")

        # now we can tokenize the conversation
        add_tokens(50256, 0)  # <|end_of_text|> following the GPT2 tokenizer
        for i, message in enumerate(messages):
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(
                    content, str
                ), "User messages are simply expected to be strings"
                value_ids = self.tokenizer.encode_ordinary(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.tokenizer.encode_ordinary(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.tokenizer.encode_ordinary(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        else:
                            raise ValueError(
                                f"Currently unsupported part type: {part['type']}"
                            )
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        """
        Used for chat completion.
        """
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]
        assert (
            messages[-1]["role"] == "assistant"
        ), "Last message must be from the Assistant"
        messages.pop()  # remove the last message (of the Assistant) inplace

        # tokenize
        ids, mask = self.render_conversation(conversation)

        # Append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """visualize for debugging"""
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        GRAY = "\033[90m"
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.tokenizer.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)


def _check(tok, name, convo):
    ids, mask = tok.render_conversation(convo)
    print(f"\n=== {name} ===")
    print(tok.visualize_tokenization(ids, mask))
    assert len(ids) == len(mask), "ids/mask length mismatch"
    # what the model is actually trained to produce (mask == 1)
    trained = tok.tokenizer.decode([t for t, m in zip(ids, mask) if m == 1])
    print("TRAINED-ON:", repr(trained))
    assert mask[0] == 0, "BOS must be mask 0"
    return ids, mask


def main():
    tok = ChatTokenizer()

    # 1) toy 2-turn: only assistant content + <|assistant_end|> should be green
    ids, mask = _check(
        tok,
        "toy 2-turn",
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "It is 4."},
            ]
        },
    )
    trained = tok.tokenizer.decode([t for t, m in zip(ids, mask) if m == 1])
    assert trained.startswith("It is 4."), trained
    assert "<|assistant_end|>" in trained, trained

    # 2) system prompt: exercises the merge branch (system text ends up in the
    #    red user turn; must not crash)
    _check(
        tok,
        "system prompt",
        {
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey."},
            ]
        },
    )

    # 3) a REAL SmolTalk row (multi-turn); loaded here, not at import time
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    _check(tok, "real smoltalk row", ds[0])  # ds[0] is already {"messages": [...]}

    print("\nall checks passed")


if __name__ == "__main__":
    main()
