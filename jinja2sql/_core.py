from __future__ import annotations

import contextlib
import inspect
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, Callable, Protocol, TypeVar, Union, overload

import jinja2
import jinja2.defaults
import jinja2.nodes
from jinja2.ext import Extension
from jinja2.lexer import Token, TokenStream
from jinja2.parser import Parser
from markupsafe import Markup
from typing_extensions import Literal, ParamSpec, TypeAlias

_T_co = TypeVar("_T_co", covariant=True)
T = TypeVar("T")
P = ParamSpec("P")


class ParamStyleFunc(Protocol):
    def __call__(self, param_key: str, param_index: int) -> str: ...


ParamStyle = Literal["named", "qmark", "format", "numeric", "pyformat", "asyncpg"]
Params: TypeAlias = Union[Mapping[str, Any], Sequence[Any]]
Context: TypeAlias = Mapping[str, Any]

DEFAULT_IDENTIFIER_QUOTE_CHAR = ""
DEFAULT_PARAM_STYLE: ParamStyle = "named"


class RenderContext:
    __slots__ = (
        "param_style",
        "identifier_quote_char",
        "_params",
        "_param_index",
    )

    def __init__(
        self,
        param_style: ParamStyle | ParamStyleFunc,
        identifier_quote_char: str,
    ) -> None:
        self._params: list[Any] | dict[str, Any] = {}
        if _is_positional_param_style(param_style):
            self._params = []
        self._param_index: int = 0
        self.param_style = param_style
        self.identifier_quote_char = identifier_quote_char

    @property
    def params(self) -> Params:
        """Get the parameters."""
        return self._params

    def bind_param(
        self, name: str, value: Any, *, in_clause: bool = False
    ) -> tuple[str, int]:
        """Bind a parameter."""
        if jinja2.is_undefined(value):
            raise jinja2.UndefinedError(f"Undefined parameter '{name}' used in query.")
        self._param_index += 1
        if in_clause:
            param_key_suffix = f"__in__{self._param_index}"
        else:
            param_key_suffix = f"__{self._param_index}"
        param_key = f"{name.replace('.', '__')}{param_key_suffix}"
        if isinstance(self._params, dict):
            self._params[param_key] = value
        else:
            self._params.append(value)
        return param_key, self._param_index


class Jinja2SQL:
    def __init__(
        self,
        searchpath: str
        | os.PathLike[str]
        | Sequence[str | os.PathLike[str]]
        | None = None,
        block_start_string: str = jinja2.defaults.BLOCK_START_STRING,
        block_end_string: str = jinja2.defaults.BLOCK_END_STRING,
        variable_start_string: str = jinja2.defaults.VARIABLE_START_STRING,
        variable_end_string: str = jinja2.defaults.VARIABLE_END_STRING,
        comment_start_string: str = jinja2.defaults.COMMENT_START_STRING,
        comment_end_string: str = jinja2.defaults.COMMENT_END_STRING,
        line_statement_prefix: str | None = jinja2.defaults.LINE_STATEMENT_PREFIX,
        line_comment_prefix: str | None = jinja2.defaults.LINE_COMMENT_PREFIX,
        trim_blocks: bool = jinja2.defaults.TRIM_BLOCKS,
        lstrip_blocks: bool = jinja2.defaults.LSTRIP_BLOCKS,
        newline_sequence: Literal[
            "\n", "\r\n", "\r"
        ] = jinja2.defaults.NEWLINE_SEQUENCE,
        keep_trailing_newline: bool = jinja2.defaults.KEEP_TRAILING_NEWLINE,
        optimized: bool = True,
        finalize: Callable[..., Any] | None = None,
        cache_size: int = 400,
        auto_reload: bool = True,
        bytecode_cache: jinja2.BytecodeCache | None = None,
        enable_async: bool = False,
        param_style: ParamStyle = DEFAULT_PARAM_STYLE,
        identifier_quote_char: str = DEFAULT_IDENTIFIER_QUOTE_CHAR,
    ):
        # Set the Jinja loader
        loader: jinja2.FileSystemLoader | None = None
        if searchpath:
            loader = jinja2.FileSystemLoader(searchpath=searchpath)

        self.param_style = param_style
        self.identifier_quote_char = identifier_quote_char

        # Set the Jinja environment
        self._env = jinja2.Environment(
            block_start_string=block_start_string,
            block_end_string=block_end_string,
            variable_start_string=variable_start_string,
            variable_end_string=variable_end_string,
            comment_start_string=comment_start_string,
            comment_end_string=comment_end_string,
            line_statement_prefix=line_statement_prefix,
            line_comment_prefix=line_comment_prefix,
            trim_blocks=trim_blocks,
            lstrip_blocks=lstrip_blocks,
            newline_sequence=newline_sequence,
            keep_trailing_newline=keep_trailing_newline,
            extensions=(Jinja2SQLExtension,),
            optimized=optimized,
            finalize=finalize,
            autoescape=True,
            cache_size=cache_size,
            auto_reload=auto_reload,
            bytecode_cache=bytecode_cache,
            enable_async=enable_async,
            loader=loader,
        )

        # Default filters
        self._env.filters["bind"] = self.bind
        self._env.filters["_bind_in"] = self.bind_in_clause
        # 'inclause' will be replaced with '_bind_in' in the filter stream
        self._env.filters["inclause"] = lambda value: value
        self._env.filters["identifier"] = self.identifier

        # Set the context variable
        self._render_context_var: ContextVar[RenderContext] = ContextVar(
            "render_context"
        )

    @property
    def env(self) -> jinja2.Environment:
        """Get the Jinja environment."""
        return self._env

    def register_filter(self, name: str, func: Callable[..., Any]) -> None:
        """Register a filter."""
        has_self = any(
            param
            for param in inspect.signature(func).parameters.values()
            if param.annotation is type(self)
        )

        if has_self:
            self._env.filters[name] = lambda *args, **kwargs: func(
                self, *args, **kwargs
            )
        else:
            self._env.filters[name] = func

    @overload
    def filter(self, func: Callable[P, T]) -> Callable[P, T]: ...

    @overload
    def filter(
        self, *, name: str | None = None
    ) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

    def filter(
        self,
        func: Callable[P, T] | None = None,
        *,
        name: str | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]] | Callable[P, T]:
        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            _name = name or func.__name__
            self.register_filter(_name, func)
            return func

        if func is None:
            return decorator

        return decorator(func)

    def from_file(
        self,
        name: str | jinja2.Template,
        *,
        context: Context | None = None,
        param_style: ParamStyle | ParamStyleFunc | None = None,
        identifier_quote_char: str | None = None,
    ) -> tuple[str, Params]:
        """Load a template from a file."""
        with self._begin_render_context(
            param_style=param_style,
            identifier_quote_char=identifier_quote_char,
        ):
            template = self.env.get_template(name)
            return self._render(template, context)

    def from_string(
        self,
        source: str | jinja2.nodes.Template,
        *,
        context: Context | None = None,
        param_style: ParamStyle | ParamStyleFunc | None = None,
        identifier_quote_char: str | None = None,
    ) -> tuple[str, Params]:
        """Load a template from a string."""
        with self._begin_render_context(
            param_style=param_style,
            identifier_quote_char=identifier_quote_char,
        ):
            template = self.env.from_string(source)
            return self._render(template, context)

    async def from_file_async(
        self,
        name: str | jinja2.Template,
        *,
        context: Context | None = None,
        param_style: ParamStyle | ParamStyleFunc | None = None,
        identifier_quote_char: str | None = None,
    ) -> tuple[str, Params]:
        """Load a template from a file asynchronously."""
        with self._begin_render_context(
            param_style=param_style,
            identifier_quote_char=identifier_quote_char,
        ):
            template = self.env.get_template(name)
            return await self._render_async(template, context)

    async def from_string_async(
        self,
        source: str | jinja2.nodes.Template,
        *,
        context: Context | None = None,
        param_style: ParamStyle | ParamStyleFunc | None = None,
        identifier_quote_char: str | None = None,
    ) -> tuple[str, Params]:
        """Load a template from a string asynchronously."""
        with self._begin_render_context(
            param_style=param_style,
            identifier_quote_char=identifier_quote_char,
        ):
            template = self.env.from_string(source)
            return await self._render_async(template, context)

    @contextlib.contextmanager
    def _begin_render_context(
        self,
        param_style: ParamStyle | ParamStyleFunc | None = None,
        identifier_quote_char: str | None = None,
    ) -> Iterator[None]:
        """Begin a render context."""
        token = self._render_context_var.set(
            RenderContext(
                param_style=param_style or self.param_style,
                identifier_quote_char=identifier_quote_char
                or self.identifier_quote_char,
            )
        )
        try:
            yield
        finally:
            self._render_context_var.reset(token)

    def _render(
        self, template: jinja2.Template, context: Context | None
    ) -> tuple[str, Params]:
        """Render a template."""
        query = template.render(context or {})
        render_context = self._render_context_var.get()
        return query, render_context.params

    async def _render_async(
        self, template: jinja2.Template, context: Context | None
    ) -> tuple[str, Params]:
        """Render a template asynchronously."""
        query = await template.render_async(context or {})
        render_context = self._render_context_var.get()
        return query, render_context.params

    def _bind_param(self, key: str, value: Any, *, in_clause: bool = False) -> str:
        """Bind a parameter."""
        render_context = self._render_context_var.get()
        param_key, param_index = render_context.bind_param(
            key, value, in_clause=in_clause
        )
        if callable(param_style := render_context.param_style):
            return param_style(param_key, param_index)
        elif param_style == "named":
            return f":{param_key}"
        elif param_style == "qmark":
            return "?"
        elif param_style == "format":
            return "%s"
        elif param_style == "numeric":
            return f":{param_index}"
        elif param_style == "pyformat":
            return f"%({param_key})s"
        elif param_style == "asyncpg":
            return f"${param_index}"
        raise ValueError(f"Invalid param_style - {param_style}")

    def bind(self, value: Any, name: str) -> Markup | str:
        """Bind a parameter."""
        if isinstance(value, Markup):
            return value
        return self._bind_param(name, value)

    def bind_in_clause(self, value: Any, name: str) -> str:
        """Bind an IN clause."""
        values = list(value)
        results = []
        for item in values:
            results.append(self._bind_param(name, item, in_clause=True))
        return f"({', '.join(results)})"

    def identifier(self, value: Any) -> Markup:
        """Format an identifier."""
        if isinstance(value, str):
            identifier = (value,)
        else:
            identifier = value
        if not isinstance(value, Iterable):
            raise ValueError("identifier filter expects a string or an Iterable")

        def _quote_and_escape(item: str) -> str:
            render_context = self._render_context_var.get()
            identifier_quote_char = render_context.identifier_quote_char
            return "".join(
                [
                    identifier_quote_char,
                    item.replace(
                        identifier_quote_char,
                        identifier_quote_char * 2,
                    ),
                    identifier_quote_char,
                ]
            )

        return Markup(".".join(_quote_and_escape(item) for item in identifier))


class Jinja2SQLExtension(Extension):
    """Jinja2SQL extension."""

    skip_filters = ("bind", "_bind_in", "safe")

    def parse(self, parser: Parser) -> jinja2.nodes.Node | list[jinja2.nodes.Node]:
        """Parse the template."""
        return []

    def filter_stream(self, stream: TokenStream) -> Iterable[Token]:
        """Filter the stream."""
        while not stream.eos:
            token = next(stream)
            if token.test("variable_begin"):
                var_expr = []
                while not token.test("variable_end"):
                    var_expr.append(token)
                    token = next(stream)
                variable_end = token
                last_token = var_expr[-1]
                lineno = last_token.lineno
                if (
                    not last_token.test("name")
                    or last_token.value not in self.skip_filters
                ):
                    if last_token.value == "inclause":
                        filter_name = "_bind_in"
                    else:
                        filter_name = "bind"

                    param_name = self._extract_param_name(var_expr)

                    var_expr.insert(1, Token(lineno, "lparen", "("))
                    var_expr.append(Token(lineno, "rparen", ")"))
                    var_expr.append(Token(lineno, "pipe", "|"))
                    var_expr.append(Token(lineno, "name", filter_name))
                    var_expr.append(Token(lineno, "lparen", "("))
                    var_expr.append(Token(lineno, "string", param_name))
                    var_expr.append(Token(lineno, "rparen", ")"))

                var_expr.append(variable_end)
                for token in var_expr:
                    yield token
            else:
                yield token

    @staticmethod
    def _extract_param_name(tokens: list[Token]) -> str:
        """Extract the parameter name."""
        name = ""
        for token in tokens:
            if token.test("variable_begin"):
                continue
            elif token.test("name"):
                name += token.value
            elif token.test("dot"):
                name += token.value
            else:
                break
        if not name:
            name = "bind_0"
        return name


def _is_positional_param_style(param_style: ParamStyle | ParamStyleFunc) -> bool:
    """Check if the param_style is positional."""
    return (
        param_style
        in (
            "qmark",
            "format",
            "numeric",
            "asyncpg",
        )
        or callable(param_style)
        and "key" not in param_style("key", 0)
    )
