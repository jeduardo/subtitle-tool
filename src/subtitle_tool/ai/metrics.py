from dataclasses import dataclass
from threading import Lock


@dataclass
class OperationMetrics:
    """
    Usage tracker for interesting metrics.

    Args:
        input_token_count (int): number of input tokens, derived from the model prompt.
        output_token_count (int): number of output tokens, comprising both output tokens
            and thought tokens.
        client_errors (int): how many errors the client has seen
        server_errors (int): how many errors the client has seen
        throttles (int): how many errors the client has seen
        retries (int): how many retries the client has seen
        invalid_subtitles (int): how many invalid subtitles were generated
        generation_errors (int): how many malformed responses the AI returned
    """

    input_token_count: int = 0
    output_token_count: int = 0
    client_errors: int = 0
    server_errors: int = 0
    throttles: int = 0
    retries: int = 0
    invalid_subtitles: int = 0
    generation_errors: int = 0

    def __post_init__(self):
        self.lock = Lock()

    def add_metrics(
        self,
        input_token_count: int = 0,
        output_token_count: int = 0,
        client_errors: int = 0,
        server_errors: int = 0,
        throttles: int = 0,
        retries: int = 0,
        invalid_subtitles: int = 0,
        generation_errors: int = 0,
    ) -> None:
        """
        Add usage counters from the current client.
        To allow the client to be used by multple threads, updates are
        performed under a lock.

        Args:
            input_token_count (int): number of input tokens to add to the
                current counter
            output_token_count (int): number of output tokens to add to the
                current counter
            client_errors (int): number of client errors to add to the
                current counter
            server_errors (int): number of server errors to add to the
                current counter
            throttles (int): number of throttles to add to the current counter
            retries (int): number of retries to add to the current counter
            invalid_subtitles (int): number of invalid subtitles to add
            generation_errors (int): number of generation errors to add
        """
        with self.lock:
            self.input_token_count += input_token_count
            self.output_token_count += output_token_count
            self.client_errors += client_errors
            self.server_errors += server_errors
            self.throttles += throttles
            self.retries += retries
            self.invalid_subtitles += invalid_subtitles
            self.generation_errors += generation_errors
