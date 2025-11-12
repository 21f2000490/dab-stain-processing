# Attribution: https://gist.github.com/porridgewithraisins/313a26ee3b827f7338df78c72ccbb247


import argparse
from dataclasses import _MISSING_TYPE, fields
from typing import Any, ClassVar, Literal, Protocol, Type, TypeVar, get_args, get_origin, get_type_hints


# note: some of the type related imports above might need changing depending on python version


class IsDataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


T = TypeVar("T", bound=IsDataclass)


def parse_args(description: str, cls: Type[T]) -> T:
    """
    Make a dataclass and annotate it with type hints and provide defaults in dataclasses.field(). You can provide the
    description in field(metadata={"help": "here"}) and it will automatically show up in the CLI. Give a type hint of
    bool for CLI switches, and a typing.Literal (or | syntax) hint for a fixed set of choices. Any other type accepted
    by argparse also works e.g Path, argparse.FileType, ascii, ord, etc. argparse also accepts a post-proc function as
    type. If you want to use this feature, pass a function (str -> Any) in the "postprocess" key of the metadata dict.
    """
    parser = argparse.ArgumentParser(description=description)
    type_hints = get_type_hints(cls)

    for f in fields(cls):
        field_name = f.name
        type_hint = type_hints[field_name]
        default = f.default
        help_msg = f.metadata.get("help", "")
        post_processor = f.metadata.get("postprocess")
        arg_name = f"--{field_name.replace('_', '-')}"

        if type_hint is bool:
            if default is False:
                parser.add_argument(arg_name, action="store_true", help=f"{help_msg} Default disabled")
            elif default is True:
                group = parser.add_mutually_exclusive_group(required=False)
                group.add_argument(
                    arg_name, dest=field_name, action="store_true", help=f"Enable {help_msg.lower()} Default enabled."
                )
                group.add_argument(
                    f"--no-{field_name.replace('_', '-')}",
                    dest=field_name,
                    action="store_false",
                    help=f"Disable {help_msg.lower()} Default enabled.",
                )
                parser.set_defaults(**{field_name: default})
            else:
                raise ValueError(f"Bool field {field_name} has invalid default: {default}")
        elif get_origin(type_hint) is Literal:
            choices = get_args(type_hint)
            lit_type = type(choices[0]) if choices else str
            if not isinstance(default, _MISSING_TYPE):
                parser.add_argument(
                    arg_name,
                    type=lit_type,
                    choices=choices,
                    default=default,
                    help=help_msg,
                )
            else:
                parser.add_argument(
                    arg_name,
                    type=lit_type,
                    choices=choices,
                    required=True,
                    help=help_msg,
                )
        else:
            typeclass = post_processor if post_processor else type_hint
            parser.add_argument(arg_name, type=typeclass, default=default, help=help_msg)

    args = parser.parse_args()

    return cls(**vars(args))
