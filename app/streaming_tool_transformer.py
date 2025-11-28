import copy
import json
import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple


logger = logging.getLogger(__name__)


class ParserState(Enum):
    IDLE = "idle"
    TOOL_CALL_BUILD = "tool_call_build"
    ARGUMENT_BUILD = "argument_build"


@dataclass
class ToolCallBuilder:
    """Builder for reconstructing a tool call from streaming chunks."""
    buffer: str = ""
    tool_name: str = ""
    tool_id: str = ""
    arguments_buffer: str = ""

    brace_depth: int = 0
    in_string: bool = False
    string_char: Optional[str] = None
    escaped: bool = False

    def reset(self) -> None:
        self.buffer = ""
        self.tool_name = ""
        self.tool_id = ""
        self.arguments_buffer = ""
        self.brace_depth = 0
        self.in_string = False
        self.string_char = None
        self.escaped = False


class StreamingToolCallTransformer:
    """
    Transforms streaming chunks with embedded tool calls in reasoning_content
    into properly structured tool_calls chunks.

    Usage pattern:

        transformer = StreamingToolCallTransformer()
        for out_chunk in transformer.process_chunk(parsed_chunk):
            ...
        for pending in transformer.flush_pending():
            ...

    If reconstruction fails, flush_pending() will yield any buffered original
    chunks unchanged.
    """

    def __init__(self) -> None:
        self.state: ParserState = ParserState.IDLE
        self.builder = ToolCallBuilder()
        self.pending_chunks: List[Dict[str, Any]] = []

        # Unicode control characters used by SGlang for special markers
        self.control_pattern = re.compile(r"[\u0f00-\u0fff\u1800-\u18af]+")
        # Explicit textual markers, in case SGlang emits them verbatim
        self.marker_pattern = re.compile(
            r"<\|tool_calls_section_begin\|>"
            r"|<\|tool_call_begin\|>"
            r"|<\|tool_call_argument_begin\|>"
            r"|<\|tool_call_end\|>"
            r"|<\|tool_calls_section_end\|>"
        )
        # functions.read_file:1 style
        self.function_pattern = re.compile(r"functions\.(\w+):(\d+)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Process a single chunk and yield 0 or more transformed chunks.

        Important:
        - Only the first choice (index 0) is examined/transformed.
        - If the chunk does not fit the expected schema, it is yielded unchanged
          while idle. If we are in the middle of building a tool call, invalid
          chunks are buffered (for possible emergency flush) but NOT forwarded,
          to avoid leaking raw tool_call markers to the client.
        """
        if not self._is_valid_chunk(chunk):
            # While idle, just pass invalid chunks through unchanged.
            if self.state == ParserState.IDLE:
                yield chunk
            else:
                # In TOOL_CALL_BUILD / ARGUMENT_BUILD we are reconstructing a
                # tool call, so keep the chunk for potential flush but do not
                # forward it directly. This prevents raw marker chunks from
                # leaking to the client after we have started the tool call.
                logger.debug(
                    "StreamingToolCallTransformer: buffering invalid chunk while in state=%s; keys=%s",
                    self.state,
                    list(chunk.keys()),
                )
                self.pending_chunks.append(copy.deepcopy(chunk))
            return

        delta = chunk["choices"][0].get("delta", {})
        reasoning_content = delta.get("reasoning_content")

        # If there's no reasoning content and we're idle, just pass through
        if self.state == ParserState.IDLE and not reasoning_content:
            yield chunk
            return

        if self.state == ParserState.IDLE:
            yield from self._process_idle_state(chunk, reasoning_content)
        elif self.state == ParserState.TOOL_CALL_BUILD:
            yield from self._process_tool_call_build_state(chunk, reasoning_content)
        elif self.state == ParserState.ARGUMENT_BUILD:
            yield from self._process_argument_build_state(chunk, reasoning_content)
        else:
            # Fallback safety
            yield chunk

    def flush_pending(self) -> Iterator[Dict[str, Any]]:
        """
        Flush any pending original chunks if we ended the stream with an
        incomplete tool call. After this, the transformer is reset.
        """
        if self.pending_chunks:
            logger.warning(
                "StreamingToolCallTransformer: flushing %d pending chunks with incomplete tool call; state=%s buffer_prefix=%r args_prefix=%r",
                len(self.pending_chunks),
                self.state,
                self.builder.buffer[:80],
                self.builder.arguments_buffer[:80],
            )
            for ch in self.pending_chunks:
                yield ch
        self._reset_state()

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _process_idle_state(
        self, chunk: Dict[str, Any], reasoning_content: Optional[str]
    ) -> Iterator[Dict[str, Any]]:
        if not reasoning_content:
            # Nothing to do
            yield chunk
            return

        cleaned, has_marker = self._strip_markers_and_detect(reasoning_content)
        looks_like_header = self._looks_like_tool_start(cleaned)

        # No markers, no header pattern: pass through unchanged
        if not has_marker and not looks_like_header:
            yield chunk
            return

        # Case 1: we see a marker but no header text yet
        # Example: "I'll analyze this. <|tool_call_begin|>"
        if has_marker and "functions." not in cleaned:
            logger.debug(
                "StreamingToolCallTransformer: detected tool marker in IDLE, entering TOOL_CALL_BUILD. cleaned=%r",
                cleaned,
            )
            # Forward the cleaned reasoning as normal assistant reasoning
            # but without leaking control markers.
            new_chunk = self._clone_chunk_with_reasoning(chunk, cleaned)
            yield new_chunk

            # Enter TOOL_CALL_BUILD for upcoming header segments
            self.state = ParserState.TOOL_CALL_BUILD
            self.builder.reset()
            self.pending_chunks = []
            return

        # Case 2: we see something that looks like a header (with or without markers)
        # Split prefix reasoning vs header tail at first "functions."
        header_start = cleaned.find("functions.")
        if header_start == -1:
            # As a fallback, if we somehow ended up here without "functions.",
            # just pass through.
            yield chunk
            return

        prefix = cleaned[:header_start]
        header = cleaned[header_start:]

        # Emit prefix reasoning (if any) as a separate chunk
        if prefix.strip():
            prefix_chunk = self._clone_chunk_with_reasoning(chunk, prefix)
            yield prefix_chunk

        # Now begin building a tool call from the header tail and subsequent chunks
        self.state = ParserState.TOOL_CALL_BUILD
        self.builder.reset()
        self.builder.buffer = header
        # Keep the original chunk for possible fallback
        self.pending_chunks = [copy.deepcopy(chunk)]

        # Try to extract tool info immediately from the current buffer
        yield from self._try_finish_tool_header(chunk_from=chunk, add_to_pending=False)

    def _process_tool_call_build_state(
        self, chunk: Dict[str, Any], reasoning_content: Optional[str]
    ) -> Iterator[Dict[str, Any]]:
        # While building a header, subsequent chunks are all considered part
        # of the header buffer (after stripping markers).
        cleaned, _ = self._strip_markers_and_detect(reasoning_content or "")
        self.builder.buffer += cleaned
        # Keep original chunk for possible fallback
        self.pending_chunks.append(copy.deepcopy(chunk))

        yield from self._try_finish_tool_header(chunk_from=chunk, add_to_pending=False)

    def _try_finish_tool_header(
        self, chunk_from: Dict[str, Any], add_to_pending: bool
    ) -> Iterator[Dict[str, Any]]:
        """
        Attempt to extract functions.<name>:<id> from builder.buffer.
        If successful:
          - Move to ARGUMENT_BUILD
          - Emit a tool_call start chunk
          - Treat any trailing text after the header as the first
            argument segment to stream.
        """
        match = self.function_pattern.search(self.builder.buffer)
        if not match:
            return

        tool_name, tool_id = match.group(1), match.group(2)
        self.builder.tool_name = tool_name
        self.builder.tool_id = tool_id

        header_end = match.end()
        # Anything after the header becomes the initial argument segment.
        remaining = self.builder.buffer[header_end:]
        # Clear header buffer once we've extracted tool info
        self.builder.buffer = ""

        # Emit the tool call start chunk
        start_chunk = self._create_tool_call_start_chunk(
            tool_name, tool_id, chunk_from
        )
        logger.debug(
            "StreamingToolCallTransformer: detected tool header functions.%s:%s",
            tool_name,
            tool_id,
        )
        yield start_chunk

        # Switch to argument build state
        self.state = ParserState.ARGUMENT_BUILD

        # If there is any tail content after the header, treat it as the first
        # argument segment (may still contain markers).
        if remaining:
            for out_chunk in self._emit_argument_delta(remaining, chunk_from):
                yield out_chunk

    def _process_argument_build_state(
        self, chunk: Dict[str, Any], reasoning_content: Optional[str]
    ) -> Iterator[Dict[str, Any]]:
        # Keep the original for possible emergency flush.
        self.pending_chunks.append(copy.deepcopy(chunk))

        # Stream arguments as tool_calls deltas.
        for out_chunk in self._emit_argument_delta(reasoning_content or "", chunk):
            yield out_chunk

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strip_markers_and_detect(self, text: str, strip_whitespace: bool = True) -> Tuple[str, bool]:
        """
        Remove known SGlang control markers (Unicode range and explicit
        ASCII markers) and return (cleaned_text, has_any_marker).

        If strip_whitespace is True, leading/trailing whitespace is stripped.
        For tool call arguments we set strip_whitespace=False to preserve
        spaces that may be split across chunks.
        """
        if not text:
            return "", False

        has_control = bool(self.control_pattern.search(text))
        no_control = self.control_pattern.sub("", text)

        has_ascii_marker = bool(self.marker_pattern.search(no_control))
        cleaned = self.marker_pattern.sub("", no_control)

        if strip_whitespace:
            cleaned = cleaned.strip()

        return cleaned, bool(has_control or has_ascii_marker)

    def _strip_argument_markers(self, text: str) -> str:
        """
        Remove only the explicit SGlang ASCII marker tokens from a segment
        that belongs to tool call arguments. We intentionally do NOT strip
        whitespace or any other characters here (including backslashes),
        to keep the raw argument payload as close to original as possible.
        """
        if not text:
            return ""
        for marker in (
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ):
            text = text.replace(marker, "")
        return text

    def _emit_argument_delta(
        self, raw_text: str, original_chunk: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """
        Given a raw reasoning_content segment that belongs to the current
        tool call's arguments, emit an incremental tool_calls delta.

        - Strips SGlang markers (including argument_begin / end markers).
        - Appends cleaned text to builder.arguments_buffer.
        - Emits a chunk where function.arguments = the new segment only.
        - If we see tool_call_end / tool_calls_section_end markers in the raw
          text, or detect a complete JSON object, reset back to IDLE.
        """
        if not raw_text:
            return

        # For arguments, avoid any sanitation beyond removing explicit markers
        # so that backslashes, quotes, and whitespace are preserved exactly
        # as emitted by the model.
        cleaned_args = self._strip_argument_markers(raw_text)

        if cleaned_args:
            self.builder.arguments_buffer += cleaned_args
            logger.debug(
                "StreamingToolCallTransformer: streaming args for %s:%s segment=%r",
                self.builder.tool_name,
                self.builder.tool_id,
                cleaned_args,
            )

            new_chunk = copy.deepcopy(original_chunk)
            new_chunk["choices"][0]["delta"] = {
                "role": None,
                "content": None,
                "reasoning_content": None,
                "tool_calls": [
                    {
                        "id": f"functions.{self.builder.tool_name}:{self.builder.tool_id}",
                        "index": 0,
                        "type": "function",
                        "function": {
                            "name": None,
                            "arguments": cleaned_args,
                        },
                    }
                ],
            }
            yield new_chunk

            # Optional: if there is no explicit end marker but the aggregated
            # arguments_buffer now forms a complete JSON object, reset state.
            # This prevents us from accidentally swallowing later reasoning.
            complete, _ = self._try_parse_json(self.builder.arguments_buffer)
            if complete and self.state == ParserState.ARGUMENT_BUILD:
                logger.debug(
                    "StreamingToolCallTransformer: detected complete JSON arguments for %s:%s",
                    self.builder.tool_name,
                    self.builder.tool_id,
                )
                self._reset_state()
                return

        # Detect end markers in the raw text (before stripping)
        if (
            "<|tool_call_end|>" in raw_text
            or "<|tool_calls_section_end|>" in raw_text
        ):
            logger.debug(
                "StreamingToolCallTransformer: detected end marker for %s:%s",
                self.builder.tool_name,
                self.builder.tool_id,
            )
            self._reset_state()

    def _looks_like_tool_start(self, text: str) -> bool:
        """Heuristic check for text that looks like the start of a tool call."""
        if not text:
            return False
        if "functions." in text:
            return True
        if self.function_pattern.search(text):
            return True
        return False

    def _try_parse_json(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Detect when we have a complete top-level JSON object in `text`.
        We track brace depth and string/escape state and, when depth
        returns to zero, attempt json.loads to confirm validity.

        Returns (is_complete, json_string_or_none).
        """
        if not text:
            return False, None

        # Walk the buffer incrementally
        for i, ch in enumerate(text):
            if self.builder.in_string:
                if self.builder.escaped:
                    self.builder.escaped = False
                elif ch == "\\":
                    self.builder.escaped = True
                elif ch == self.builder.string_char:
                    self.builder.in_string = False
                    self.builder.string_char = None
            else:
                if ch in ('"', "'"):
                    self.builder.in_string = True
                    self.builder.string_char = ch
                elif ch == "{":
                    if self.builder.brace_depth == 0:
                        self.builder.brace_depth = 1
                    else:
                        self.builder.brace_depth += 1
                elif ch == "}":
                    if self.builder.brace_depth > 0:
                        self.builder.brace_depth -= 1
                        if self.builder.brace_depth == 0:
                            candidate = text[: i + 1]
                            try:
                                json.loads(candidate)
                                return True, candidate
                            except json.JSONDecodeError:
                                # Not valid yet, continue scanning
                                pass

        return False, None

    def _clone_chunk_with_reasoning(
        self, original_chunk: Dict[str, Any], reasoning: str
    ) -> Dict[str, Any]:
        """Clone a chunk and override its reasoning_content with `reasoning`."""
        new_chunk = copy.deepcopy(original_chunk)
        delta = new_chunk["choices"][0].setdefault("delta", {})
        delta["reasoning_content"] = reasoning
        # Ensure we don't accidentally forward any tool_calls with this piece
        if "tool_calls" in delta:
            delta["tool_calls"] = None
        return new_chunk

    def _create_tool_call_start_chunk(
        self, tool_name: str, tool_id: str, original_chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create the initial tool call announcement chunk:

        {
          "delta": {
            "role": null,
            "content": null,
            "reasoning_content": null,
            "tool_calls": [
              {
                "id": "functions.read_file:1",
                "index": 0,
                "type": "function",
                "function": {
                  "name": "read_file",
                  "arguments": ""
                }
              }
            ]
          }
        }
        """
        new_chunk = copy.deepcopy(original_chunk)
        new_chunk["choices"][0]["delta"] = {
            "role": None,
            "content": None,
            "reasoning_content": None,
            "tool_calls": [
                {
                    "id": f"functions.{tool_name}:{tool_id}",
                    "index": 0,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": "",
                    },
                }
            ],
        }
        return new_chunk

    def _create_tool_call_argument_chunk(
        self, tool_name: str, tool_id: str, args_json: str, original_chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create the chunk containing the tool call arguments as a single
        complete JSON string in `function.arguments`.
        """
        new_chunk = copy.deepcopy(original_chunk)
        new_chunk["choices"][0]["delta"] = {
            "role": None,
            "content": None,
            "reasoning_content": None,
            "tool_calls": [
                {
                    "id": f"functions.{tool_name}:{tool_id}",
                    "index": 0,
                    "type": "function",
                    "function": {
                        "name": None,  # Name already sent
                        "arguments": args_json,
                    },
                }
            ],
        }
        return new_chunk

    def _is_valid_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Check if chunk is a valid streaming chunk we know how to handle."""
        if not isinstance(chunk, dict):
            return False
        choices = chunk.get("choices")
        if not choices or not isinstance(choices, list):
            return False
        first = choices[0]
        if not isinstance(first, dict):
            return False
        if "delta" not in first:
            return False
        return True

    def _reset_state(self) -> None:
        """Reset parser state and builder."""
        self.state = ParserState.IDLE
        self.builder.reset()
        self.pending_chunks = []


__all__ = ["StreamingToolCallTransformer"]
