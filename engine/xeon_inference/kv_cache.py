from dataclasses import dataclass


@dataclass
class KVCacheState:
    allocated_tokens: int = 0
    max_tokens: int = 0

    def can_allocate(self, tokens: int) -> bool:
        return self.allocated_tokens + tokens <= self.max_tokens

    def allocate(self, tokens: int) -> bool:
        if self.can_allocate(tokens):
            self.allocated_tokens += tokens
            return True
        return False

    def free(self, tokens: int) -> None:
        self.allocated_tokens = max(0, self.allocated_tokens - tokens)
