"""LLM 客户端抽象基类，定义统一的生成接口。"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator

from pydantic import BaseModel

from backend.llm.config import LLMProviderConfig
from backend.llm.generation_options import REASONING_FIELDS
from backend.llm.models import LLMFunctionCallProbe, LLMRequest, LLMResponse, LLMToolDefinition, LLMToolResult, LLMToolTurn


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def apply_request_defaults(request: LLMRequest, config: LLMProviderConfig) -> LLMRequest:
    """补齐模型服务的默认生成参数，不创建客户端。

    Args:
        request: 节点或业务已构建的请求。
        config: 模型服务配置；本函数只使用生成参数。
    Returns:
        补齐默认值后的请求，原请求保持不变。
    """
    overrides = {}
    # 单次请求改变思考方式时，不带入与新方式冲突的 Provider 默认值。
    skip_defaults = set()
    if request.thinking_mode == "disabled":
        skip_defaults.update({"reasoning_effort", "thinking_budget"})
    if request.thinking_mode == "enabled" and config.reasoning_effort == "none":
        skip_defaults.add("reasoning_effort")
    if request.thinking_budget is not None:
        skip_defaults.update({"reasoning_effort", "thinking_mode"})
    if request.reasoning_effort is not None:
        skip_defaults.add("thinking_budget")
        if (config.thinking_mode == "disabled" or request.reasoning_effort == "none"
                or (config.type == "claude" and config.thinking_mode == "enabled")):
            skip_defaults.add("thinking_mode")
    if request.thinking_mode in {"adaptive", "between_tools"}:
        skip_defaults.add("thinking_budget")
    for field in ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", *REASONING_FIELDS):
        if field not in skip_defaults and getattr(request, field) is None and getattr(config, field) is not None:
            overrides[field] = getattr(config, field)
    if not request.system_prompt and config.system_prompt:
        overrides["system_prompt"] = config.system_prompt
    return request.model_copy(update=overrides) if overrides else request


def build_request_messages(request: LLMRequest) -> list[dict[str, str]]:
    """合并系统提示与对话消息，供正式请求和模拟查看共同使用。

    Args:
        request: 已渲染并补齐默认值的请求。
    Returns:
        完整消息列表。
    """
    messages = [{"role": "system", "content": request.system_prompt}] if request.system_prompt else []
    return [*messages, *request.messages]


class _ThinkStreamFilter:
    """流式文本过滤器，支持跨 chunk 移除 <think>...</think> 内容。"""

    def __init__(self, *, preserve_json_strings: bool = False) -> None:
        self._buffer = ""
        self._inside_think = False
        self._preserve_json_strings = preserve_json_strings
        self._inside_json_string = False
        self._json_escaped = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""

        self._buffer += chunk
        output_parts: list[str] = []

        while self._buffer:
            if self._inside_think:
                close_match = _THINK_CLOSE_RE.search(self._buffer)
                if close_match:
                    self._buffer = self._buffer[close_match.end():]
                    self._inside_think = False
                    continue

                keep_len = len("</think>") - 1
                if len(self._buffer) > keep_len:
                    self._buffer = self._buffer[-keep_len:]
                break

            if self._inside_json_string:
                # JSON 字符串里的标签是字段内容；转义状态需要跨片段保留。
                position = 0
                for character in self._buffer:
                    position += 1
                    if self._json_escaped:
                        self._json_escaped = False
                    elif character == "\\":
                        self._json_escaped = True
                    elif character == '"':
                        self._inside_json_string = False
                        break
                output_parts.append(self._buffer[:position])
                self._buffer = self._buffer[position:]
                continue

            open_match = _THINK_OPEN_RE.search(self._buffer)
            close_match = _THINK_CLOSE_RE.search(self._buffer) if self._preserve_json_strings else None
            tag_match = open_match
            if close_match is not None and (open_match is None or close_match.start() < open_match.start()):
                tag_match = close_match
            quote = self._buffer.find('"') if self._preserve_json_strings else -1
            # 未收完的标签可能含属性引号，必须等标签结束后再判断 JSON 字符串。
            if 0 <= quote < self._safe_emit_length(self._buffer) and (tag_match is None or quote < tag_match.start()):
                output_parts.append(self._buffer[:quote + 1])
                self._buffer = self._buffer[quote + 1:]
                self._inside_json_string = True
                continue
            if tag_match:
                if tag_match.start() > 0:
                    output_parts.append(self._buffer[:tag_match.start()])
                self._buffer = self._buffer[tag_match.end():]
                self._inside_think = tag_match is open_match
                continue

            safe_length = self._safe_emit_length(self._buffer)
            if safe_length <= 0:
                break
            output_parts.append(self._buffer[:safe_length])
            self._buffer = self._buffer[safe_length:]

        return "".join(output_parts)

    def finish(self) -> str:
        if self._inside_think:
            return ""
        if self._is_potential_think_prefix(self._buffer):
            return ""
        return BaseLLMClient._sanitize_text_content(self._buffer)

    def _safe_emit_length(self, text: str) -> int:
        if not self._preserve_json_strings:
            last_lt = text.rfind("<")
            return last_lt if last_lt >= 0 and self._is_potential_think_prefix(text[last_lt:]) else len(text)
        for marker in re.finditer("<", text):
            if self._is_potential_think_prefix(text[marker.start():]):
                return marker.start()
        return len(text)

    @staticmethod
    def _is_potential_think_prefix(text: str) -> bool:
        lowered = text.lower()

        if "<think".startswith(lowered) or "</think>".startswith(lowered):
            return True

        if lowered.startswith("<think"):
            if len(lowered) == len("<think"):
                return True
            next_char = lowered[len("<think")]
            return not (next_char.isalnum() or next_char == "_")

        if lowered.startswith("</think"):
            if len(lowered) == len("</think"):
                return True
            next_char = lowered[len("</think")]
            return next_char == ">"

        return False


class _BeforeSendError(Exception):
    """只穿过 SDK 的校验失败，适配器必须还原原异常。"""

    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(str(original))


class BaseLLMClient(ABC):
    """所有 LLM 服务商客户端的抽象基类。

    子类需实现 text_generate / schema_generate / stream_text 三个核心方法。
    """

    def __init__(self, config: LLMProviderConfig, provider_name: str = "") -> None:
        self.config = config
        self.provider_name = provider_name
        self.before_send: Callable[[], Awaitable[Any]] | None = None

    async def _before_http_request(self, request: Any) -> None:
        """每次真正发送前复验本次调用者的读取，包括 SDK 内部重试。

        Args:
            request: HTTP 客户端即将发送的请求，不修改请求内容。
        Returns:
            无；校验失败由适配器原样交还业务层。
        """
        if self.before_send is not None:
            try:
                await self.before_send()
            except Exception as exc:
                # 校验可能因数据库暂不可用失败，也不能让 SDK 当网络错误重发。
                raise _BeforeSendError(exc) from None

    def _resolve_model(self, request: LLMRequest) -> str:
        """确定实际使用的模型名称，请求未指定时回退到服务商默认模型。"""
        return request.model or self.config.default_model

    def _apply_defaults(self, request: LLMRequest) -> LLMRequest:
        """将服务商配置中的生成参数默认值合并到请求中（请求级别优先）。"""
        return apply_request_defaults(request, self.config)

    def _build_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        """将 system_prompt 与 messages 合并为完整消息列表。"""
        return build_request_messages(request)

    @staticmethod
    def _sanitize_text_content(text: str, *, preserve_json_strings: bool = False) -> str:
        """移除思考块，JSON 请求保留字段字符串里的标签。

        Args:
            text: 尚未过滤的模型正文。
            preserve_json_strings: 是否把 JSON 字符串中的标签当作字段内容。
        Returns:
            已过滤思考块的正文。
        """
        if not text:
            return text
        if preserve_json_strings:
            stream_filter = _ThinkStreamFilter(preserve_json_strings=True)
            return (stream_filter.feed(text) + stream_filter.finish()).strip()

        sanitized = _THINK_BLOCK_RE.sub("", text)
        sanitized = _THINK_OPEN_RE.sub("", sanitized)
        sanitized = _THINK_CLOSE_RE.sub("", sanitized)

        # 一劳永逸：用正则一把梭干掉所有 LaTeX 公式（不管完整不完整、不管有没有 $ 包裹、不管流式输出时被拆成几段）
        # 1. 先把常见的 LaTeX 命令替换成对应的 Unicode 符号（完整匹配时）
        _latex_cmd_re = re.compile(r'\\(rightarrow|leftarrow|Rightarrow|Leftarrow|times|div|pm|leq|geq|neq|approx|infty|alpha|beta|gamma|delta|mu|pi|sigma|theta|lambda|omega|Delta|sum)')
        _latex_cmd_map = {
            'rightarrow': '→', 'leftarrow': '←', 'Rightarrow': '⇒', 'Leftarrow': '⇐',
            'times': '×', 'div': '÷', 'pm': '±', 'leq': '≤', 'geq': '≥',
            'neq': '≠', 'approx': '≈', 'infty': '∞',
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'mu': 'μ',
            'pi': 'π', 'sigma': 'σ', 'theta': 'θ', 'lambda': 'λ', 'omega': 'ω',
            'Delta': 'Δ', 'sum': '∑',
        }
        sanitized = _latex_cmd_re.sub(lambda m: _latex_cmd_map.get(m.group(1), ''), sanitized)
        # 2. 把所有残留的 $ 符号直接删掉（流式输出时公式被拆成两半的情况）
        sanitized = sanitized.replace('$', '')
        # 3. 把所有残留的 \ 后面跟字母的组合也删掉（不完整的 LaTeX 命令，比如 \ight、\arrow 等）
        sanitized = re.sub(r'\\[a-zA-Z]+', '', sanitized)
        # 4. 把所有残留的 LaTeX 大括号也删掉（\text{...} 被删掉命令后残留的 {}）
        sanitized = sanitized.replace('{', '').replace('}', '')
        # 5. 干掉所有非中文/非英文/非常用标点的特殊字符（阿拉伯文、希伯来文、泰文等 gemma 蹦出来的多语言垃圾）
        # 只保留：中文（\u4e00-\u9fff）、英文大小写字母、数字、常用中文标点和英文标点
        sanitized = re.sub(r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s\.,!?;:\'"()\[\]{}\-—…·、。，！？；：""''（）【】]', '', sanitized)

        # 容错：gemma 模型偶尔蹦出英文动作/表情词 → 替换成中文（避免误伤专有名词，只替换最常见的几个）
        _english_action_map = {
            " swagger ": " 大摇大摆地走 ",
            " strut ": " 趾高气扬地走 ",
            " saunter ": " 悠闲地漫步 ",
            " stroll ": " 漫步 ",
            " march ": " 行进 ",
            " trudge ": " 步履沉重地走 ",
            " plod ": " 沉重缓慢地走 ",
            " limp ": " 一瘸一拐地走 ",
            " stagger ": " 蹒跚 ",
            " stumble ": " 绊倒 ",
            " frown ": " 皱眉 ",
            " grin ": " 咧嘴笑 ",
            " smirk ": " 得意地笑 ",
            " nod ": " 点头 ",
            " shake head ": " 摇头 ",
            " shrug ": " 耸肩 ",
            " sigh ": " 叹气 ",
            " gasp ": " 倒吸一口气 ",
            " whisper ": " 低声说 ",
            " mutter ": " 咕哝 ",
            " growl ": " 低吼 ",
            " snarl ": " 咆哮 ",
            " crawled ": " 爬出来 ",
            " crept ": " 蹑手蹑脚地走 ",
            " darted ": " 飞奔 ",
            " dashed ": " 猛冲 ",
            " rushed ": " 冲 ",
            " sprinted ": " 冲刺 ",
            " leaped ": " 跳 ",
            " jumped ": " 跳 ",
            " dove ": " 俯冲 ",
            " climbed ": " 爬 ",
            " crawled out ": " 从……爬出来 ",
        }
        for _src, _dst in _english_action_map.items():
            sanitized = sanitized.replace(_src, _dst)

        # 一劳永逸：自动识别夹在中文里的英文动词（ed/ing 结尾）并删除——这种几乎肯定是 gemma 蹦出来的，不是专有名词
        # 例："他 crawled 出来" → "他出来"；"她 smiling 地说" → "她地说"
        sanitized = re.sub(r'(?<=[\u4e00-\u9fff])\s*\b[a-zA-Z]+(ed|ing)\b\s*(?=[\u4e00-\u9fff])', '', sanitized)
        # 再处理一次：英文动词后面跟中文的"地"字（"smiling 地" → "地"）
        sanitized = re.sub(r'\b[a-zA-Z]+(ed|ing)\b\s*(?=地)', '', sanitized)

        if sanitized != text:
            return sanitized.strip()
        return sanitized

    def _finalize_response(self, response: LLMResponse, *, preserve_json_strings: bool = False) -> LLMResponse:
        """只过滤业务正文，raw_response 保留服务商原始内容。

        Args:
            response: 尚未处理的统一响应。
            preserve_json_strings: JSON 请求保留字段字符串里的标签。

        Returns:
            业务正文已过滤、原始响应完整保留的对象。
        """
        clean_content = self._sanitize_text_content(response.content, preserve_json_strings=preserve_json_strings)
        if clean_content == response.content:
            return response
        return response.model_copy(update={"content": clean_content})

    def _validate_probe_arguments(
        self,
        actual_arguments: dict[str, Any],
        expected_arguments: dict[str, Any],
    ) -> None:
        """校验 Function Calling 探测返回的工具参数。

        Args:
            actual_arguments: 模型实际生成的工具参数。
            expected_arguments: 探测流程要求的工具参数。

        Returns:
            无。

        Raises:
            ValueError: 参数缺失或值不一致时抛出。
        """
        for key, expected_value in expected_arguments.items():
            if actual_arguments.get(key) != expected_value:
                raise ValueError(f"工具参数不匹配: {key}")

    def _validate_probe_final_text(self, content: str, expected_final_text: str) -> None:
        """校验 Function Calling 二轮结果是否包含约定文本。

        Args:
            content: 模型第二轮最终回复文本。
            expected_final_text: 必须出现在回复中的约定文本。

        Returns:
            无。

        Raises:
            ValueError: 回复为空或未包含约定文本时抛出。
        """
        if expected_final_text not in content:
            raise ValueError("工具结果二轮回复未包含约定文本")

    @abstractmethod
    async def text_generate(self, request: LLMRequest) -> LLMResponse:
        """普通文本生成。"""

    @abstractmethod
    async def schema_generate(
        self, request: LLMRequest, schema: type[BaseModel]
    ) -> LLMResponse:
        """结构化 JSON 输出，按 Pydantic Schema 约束生成。"""

    @abstractmethod
    async def stream_text(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """流式文本输出，逐块 yield 生成内容。"""

    @abstractmethod
    async def function_call_probe(
        self,
        request: LLMRequest,
        probe: LLMFunctionCallProbe,
    ) -> LLMResponse:
        """Function Calling 探测，要求模型调用工具并完成工具结果二轮回复。"""

    async def tool_turn(
        self, request: LLMRequest, tools: list[LLMToolDefinition], *,
        history: list[dict[str, Any]] | None = None,
        results: list[LLMToolResult] | None = None,
    ) -> LLMToolTurn:
        """执行一轮原生工具对话，具体客户端必须保留完整续接信息。

        Args:
            request: 固定的初始提示与生成参数。
            tools: 允许模型调用的工具定义。
            history: 上轮客户端返回的原生历史。
            results: 对上轮全部工具调用的执行结果。
        Returns:
            本轮调用和仅供服务端续接的历史。
        """
        raise NotImplementedError("当前模型客户端不支持正式工具调用")
