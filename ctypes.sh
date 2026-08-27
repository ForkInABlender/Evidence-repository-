#!/usr/bin/env python3
"""
ctypes.sh - pure CPython reimplementation of the useful ctypes.sh interface.

This file intentionally contains no Bash, subprocesses, libffi dependency,
compiled extension, or external helper executable.  Native calls are made
through CPython's standard-library ctypes module.

Command-compatible operations:
    dlopen, dlclose, dlsym, dlcall, callback, pack, unpack, sizeof

Python API:
    CTypesRuntime, CTypeValue, NativePointer, CStruct, CUnion

The original ctypes.sh is a Bash plugin backed by libffi.  This implementation
moves the same FFI model into Python/ctypes.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import platform
import struct as _struct
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Primitive type system
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "void": None,
    "bool": ctypes.c_int,
    "boolean": ctypes.c_int,

    "char": ctypes.c_byte,
    "uchar": ctypes.c_ubyte,
    "byte": ctypes.c_ubyte,

    "short": ctypes.c_short,
    "ushort": ctypes.c_ushort,

    "int": ctypes.c_int,
    "unsigned": ctypes.c_uint,

    "long": ctypes.c_long,
    "ulong": ctypes.c_ulong,

    "longlong": ctypes.c_longlong,

    "int8": ctypes.c_int8,
    "uint8": ctypes.c_uint8,
    "int16": ctypes.c_int16,
    "uint16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,

    "float": ctypes.c_float,
    "double": ctypes.c_double,
    "longdouble": ctypes.c_longdouble,

    "pointer": ctypes.c_void_p,
    "string": ctypes.c_char_p,
}

_POINTER_BITS = ctypes.sizeof(ctypes.c_void_p) * 8


def _ctype_for(name: str):
    key = name.strip().lower()
    if key not in _TYPE_MAP:
        raise ValueError(f"unknown ctypes.sh type: {name}")
    return _TYPE_MAP[key]


def _split_prefix(value: str) -> tuple[str | None, str]:
    if ":" not in value:
        return None, value
    a, b = value.split(":", 1)
    return a, b


def _parse_int(text: str, signed: bool = True) -> int:
    text = text.strip()
    if text.lower() in {"(nil)", "(null)", "null", "none"}:
        return 0
    return int(text, 0)


def _parse_float(text: str) -> float:
    return float(text.strip())


def _parse_pointer(text: str) -> int:
    text = text.strip()
    if text.lower() in {"(nil)", "(null)", "null", "none"}:
        return 0
    return int(text, 0)


def _parse_value(type_name: str, text: str):
    t = type_name.lower()
    if t in {"string"}:
        return ctypes.c_char_p(text.encode())
    if t == "pointer":
        return ctypes.c_void_p(_parse_pointer(text))
    if t in {"float", "double", "longdouble"}:
        return _ctype_for(t)(_parse_float(text))
    if t in {"char", "uchar", "byte", "bool", "boolean",
              "short", "ushort", "int", "unsigned", "long", "ulong",
              "longlong", "int8", "uint8", "int16", "uint16",
              "int32", "uint32", "int64", "uint64"}:
        return _ctype_for(t)(_parse_int(text))
    if t == "void":
        return None
    raise ValueError(f"cannot parse value of type {type_name}")


def _format_scalar(type_name: str, value: Any) -> str:
    t = type_name.lower()
    if isinstance(value, ctypes._SimpleCData):
        value = value.value

    if t in {"string"}:
        if value is None:
            return "string:"
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        return f"string:{value}"

    if t == "pointer":
        n = 0 if value is None else int(value)
        return f"pointer:{n:#x}"

    if t in {"float", "double", "longdouble"}:
        return f"{t}:{float(value)}"

    if t in {"char", "uchar"}:
        try:
            return f"{t}:{chr(int(value))}"
        except ValueError:
            return f"{t}:{int(value)}"

    return f"{t}:{int(value)}"


# ---------------------------------------------------------------------------
# Typed values
# ---------------------------------------------------------------------------

@dataclass
class CTypeValue:
    type_name: str
    value: Any

    @classmethod
    def parse(cls, token: str) -> "CTypeValue":
        prefix, payload = _split_prefix(token)
        if prefix is None:
            # ctypes.sh interprets legal integers as int, everything else as
            # a string when no explicit prefix is supplied.
            try:
                n = int(payload, 0)
            except ValueError:
                return cls("string", payload)
            return cls("int", n)
        if prefix.lower() not in _TYPE_MAP:
            raise ValueError(f"unknown type prefix: {prefix}")
        if prefix.lower() == "void":
            return cls("void", None)
        return cls(prefix.lower(), _parse_value(prefix, payload))

    def cvalue(self):
        if self.type_name == "void":
            return None
        return _parse_value(self.type_name, str(self.value))

    def encode(self) -> str:
        value = self.value
        if isinstance(value, ctypes._SimpleCData):
            value = value.value
        return _format_scalar(self.type_name, value)


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

class NativePointer:
    """Owns an optional ctypes allocation and exposes its native address."""

    def __init__(self, address: int = 0, owner: Any = None):
        self.address = int(address or 0)
        self.owner = owner

    def __int__(self):
        return self.address

    def __repr__(self):
        return f"pointer:{self.address:#x}"

    @property
    def value(self):
        return self.address


def alloc(size: int, zero: bool = False) -> NativePointer:
    if size < 0:
        raise ValueError("negative allocation size")
    buf = (ctypes.c_ubyte * size)()
    if not zero:
        # ctypes zero-initializes its array; this is retained because there is
        # no safe Python-level equivalent of an intentionally uninitialized
        # malloc buffer.
        pass
    return NativePointer(ctypes.addressof(buf), buf)


def read_memory(address: int, size: int) -> bytes:
    if not address:
        raise ValueError("NULL pointer")
    return ctypes.string_at(address, size)


def write_memory(address: int, data: bytes) -> None:
    if not address:
        raise ValueError("NULL pointer")
    ctypes.memmove(address, data, len(data))


# ---------------------------------------------------------------------------
# Structures/unions
# ---------------------------------------------------------------------------

_STRUCT_REGISTRY: dict[str, type[ctypes.Structure | ctypes.Union]] = {}


class CStruct(ctypes.Structure):
    """Base class for dynamically defined C structs."""
    pass


class CUnion(ctypes.Union):
    """Base class for dynamically defined C unions."""
    pass


def define_struct(name: str, fields: Sequence[tuple[str, Any]],
                  pack: int | None = None) -> type[CStruct]:
    attrs = {"_fields_": list(fields)}
    if pack is not None:
        attrs["_pack_"] = pack
    cls = type(name, (CStruct,), attrs)
    _STRUCT_REGISTRY[name] = cls
    return cls


def define_union(name: str, fields: Sequence[tuple[str, Any]],
                 pack: int | None = None) -> type[CUnion]:
    attrs = {"_fields_": list(fields)}
    if pack is not None:
        attrs["_pack_"] = pack
    cls = type(name, (CUnion,), attrs)
    _STRUCT_REGISTRY[name] = cls
    return cls


def _resolve_ctypes_type(name: str):
    name = name.strip()
    if name in _STRUCT_REGISTRY:
        return _STRUCT_REGISTRY[name]
    if name.endswith("*"):
        base = _resolve_ctypes_type(name[:-1].strip())
        return ctypes.POINTER(base)
    return _ctype_for(name)


# ---------------------------------------------------------------------------
# Runtime / dynamic linker
# ---------------------------------------------------------------------------

class CTypesRuntime:
    """
    Runtime equivalent of the ctypes.sh builtins.

    Handles are retained strongly, so loaded libraries remain alive until
    dlclose() or runtime shutdown.
    """

    NULL = 0
    STDIN_FILENO = 0
    STDOUT_FILENO = 1
    STDERR_FILENO = 2

    def __init__(self):
        self.libraries: dict[str, ctypes.CDLL] = {}
        self.handles: dict[int, ctypes.CDLL] = {}
        self.symbols: dict[str, Any] = {}
        self.callbacks: dict[int, Any] = {}
        self.values: dict[str, str] = {}

        # Python cannot portably obtain RTLD_NEXT/RTLD_DEFAULT from ctypes on
        # every platform.  CDLL(None) represents the process-global namespace.
        self.default = ctypes.CDLL(None)

    # ---- dynamic loader ---------------------------------------------------

    @staticmethod
    def _resolve_library(name: str) -> str:
        if os.path.isabs(name) or os.path.sep in name:
            return name
        found = ctypes.util.find_library(name)
        return found or name

    def dlopen(self, library: str, *flags: str, lazy: bool = True,
               global_symbols: bool = True, noload: bool = False,
               nodelete: bool = False, deepbind: bool = False):
        mode = 0

        # ctypes exposes platform dlopen constants on POSIX builds.
        mode |= getattr(os, "RTLD_LAZY", 1) if lazy else getattr(os, "RTLD_NOW", 2)
        if global_symbols:
            mode |= getattr(os, "RTLD_GLOBAL", 0)
        else:
            mode |= getattr(os, "RTLD_LOCAL", 0)

        named_flags = {f.upper() for f in flags}
        if "RTLD_LAZY" in named_flags:
            mode = (mode & ~getattr(os, "RTLD_NOW", 2)) | getattr(os, "RTLD_LAZY", 1)
        if "RTLD_NOW" in named_flags:
            mode = (mode & ~getattr(os, "RTLD_LAZY", 1)) | getattr(os, "RTLD_NOW", 2)
        if "RTLD_GLOBAL" in named_flags:
            mode |= getattr(os, "RTLD_GLOBAL", 0)
        if "RTLD_LOCAL" in named_flags:
            mode |= getattr(os, "RTLD_LOCAL", 0)

        target = self._resolve_library(library)

        # noload is not portable through ctypes.  RTLD_NOLOAD can be supplied
        # where the platform exposes it.
        if noload:
            mode |= getattr(os, "RTLD_NOLOAD", 0)
        if nodelete:
            mode |= getattr(os, "RTLD_NODELETE", 0)
        if deepbind:
            mode |= getattr(os, "RTLD_DEEPBIND", 0)

        lib = ctypes.CDLL(target, mode=mode) if mode else ctypes.CDLL(target)
        self.libraries[os.path.basename(library)] = lib
        self.handles[lib._handle] = lib
        return NativePointer(lib._handle, lib)

    def dlclose(self, handle: int | NativePointer | ctypes.CDLL):
        if isinstance(handle, ctypes.CDLL):
            lib = handle
        else:
            address = int(handle)
            lib = self.handles.get(address)
            if lib is None:
                raise ValueError(f"unknown library handle {address:#x}")

        # ctypes has no public dlclose API.  On POSIX its internal _dlclose is
        # intentionally not part of the stable Python API.  Releasing the
        # object is therefore the portable CPython operation.
        self.handles.pop(lib._handle, None)
        for k, v in list(self.libraries.items()):
            if v is lib:
                del self.libraries[k]
        del lib

    def dlsym(self, symbol: str, handle: Any = None):
        lib = self.default if handle is None else self._library_from_handle(handle)
        try:
            value = getattr(lib, symbol)
        except AttributeError:
            raise AttributeError(f"symbol not found: {symbol}")
        self.symbols[symbol] = value
        return value

    def _library_from_handle(self, handle):
        if isinstance(handle, ctypes.CDLL):
            return handle
        if isinstance(handle, NativePointer):
            handle = handle.address
        return self.handles[int(handle)]

    # ---- calls ------------------------------------------------------------

    def dlcall(self, symbol: str, *args, restype: str = "void",
               argtypes: Sequence[str] | None = None, handle: Any = None,
               name: str = "DLRETVAL"):
        fn = self.dlsym(symbol, handle)

        if argtypes is None:
            parsed = [CTypeValue.parse(str(a)) for a in args]
            argtypes = [x.type_name for x in parsed]
            cargs = [x.cvalue() for x in parsed]
        else:
            if len(argtypes) != len(args):
                raise TypeError("argtypes and args have different lengths")
            cargs = [
                _parse_value(t, str(v))
                for t, v in zip(argtypes, args)
            ]

        if restype.lower() == "void":
            fn.restype = None
        else:
            fn.restype = _ctype_for(restype)

        if argtypes:
            fn.argtypes = [_ctype_for(t) for t in argtypes]

        result = fn(*cargs)
        encoded = _format_scalar(restype, result) if restype != "void" else None
        if encoded is not None:
            self.values[name] = encoded
        return result

    # ---- callbacks --------------------------------------------------------

    def callback(self, function: Callable[..., Any],
                 return_type: str, *parameter_types: str,
                 name: str = "DLRETVAL"):
        rt = _ctype_for(return_type)
        at = [_ctype_for(x) for x in parameter_types]
        prototype = ctypes.CFUNCTYPE(rt, *at)
        cb = prototype(function)
        address = ctypes.cast(cb, ctypes.c_void_p).value or 0
        self.callbacks[address] = cb
        self.values[name] = f"pointer:{address:#x}"
        return NativePointer(address, cb)

    def release_callback(self, callback: int | NativePointer):
        address = int(callback)
        self.callbacks.pop(address, None)

    # ---- memory/array operations -----------------------------------------

    def sizeof(self, type_or_value):
        if isinstance(type_or_value, str):
            token = type_or_value.strip()
            if token in _STRUCT_REGISTRY:
                return ctypes.sizeof(_STRUCT_REGISTRY[token])
            prefix, _ = _split_prefix(token)
            if prefix:
                return ctypes.sizeof(_ctype_for(prefix))
            return ctypes.sizeof(_ctype_for(token))
        if isinstance(type_or_value, type):
            return ctypes.sizeof(type_or_value)
        return ctypes.sizeof(type_or_value)

    def pack(self, address: int | NativePointer,
             values: Sequence[str] | Mapping[str, str]):
        ptr = int(address)
        if isinstance(values, Mapping):
            values = list(values.values())
        offset = 0
        for token in values:
            cv = CTypeValue.parse(token)
            c = cv.cvalue()
            if cv.type_name == "string":
                p = ctypes.cast(c, ctypes.c_void_p).value or 0
                data = ctypes.c_void_p(p)
                size = ctypes.sizeof(data)
                ctypes.memmove(ptr + offset, ctypes.byref(data), size)
            else:
                size = ctypes.sizeof(c)
                ctypes.memmove(ptr + offset, ctypes.byref(c), size)
            offset += size
        return offset

    def unpack(self, address: int | NativePointer,
               types: Sequence[str]) -> list[str]:
        ptr = int(address)
        result = []
        offset = 0
        for token in types:
            prefix, _ = _split_prefix(token)
            if prefix is None:
                prefix = token
            ctype = _ctype_for(prefix)
            obj = ctype.from_address(ptr + offset)
            result.append(_format_scalar(prefix, obj.value))
            offset += ctypes.sizeof(ctype)
        return result


# ---------------------------------------------------------------------------
# Lightweight C declaration helper
# ---------------------------------------------------------------------------

_C_ALIASES = {
    "signed char": "int8",
    "unsigned char": "uint8",
    "char": "char",
    "short": "short",
    "unsigned short": "ushort",
    "int": "int",
    "unsigned": "unsigned",
    "unsigned int": "unsigned",
    "long": "long",
    "unsigned long": "ulong",
    "long long": "longlong",
    "unsigned long long": "uint64",
    "float": "float",
    "double": "double",
    "long double": "longdouble",
    "void": "void",
    "size_t": "uint64" if ctypes.sizeof(ctypes.c_size_t) == 8 else "uint32",
    "ssize_t": "int64" if ctypes.sizeof(ctypes.c_ssize_t) == 8 else "int32",
}


def parse_c_type(type_text: str):
    t = " ".join(type_text.replace("*", " * ").split())
    pointer = t.endswith(" *")
    if pointer:
        t = t[:-2].strip()
    t = _C_ALIASES.get(t, t)
    base = _resolve_ctypes_type(t)
    return ctypes.POINTER(base) if pointer else base


def define_c_struct(name: str, declarations: Sequence[str],
                    union: bool = False, pack: int | None = None):
    fields = []
    for declaration in declarations:
        declaration = declaration.strip().rstrip(";")
        parts = declaration.rsplit(None, 1)
        if len(parts) != 2:
            raise ValueError(f"invalid C field declaration: {declaration}")
        type_text, field = parts

        # Handle "char *name" after rsplit.
        if field.startswith("*"):
            type_text += " *"
            field = field[1:]

        fields.append((field, parse_c_type(type_text)))

    if union:
        return define_union(name, fields, pack)
    return define_struct(name, fields, pack)


# ---------------------------------------------------------------------------
# Command-line compatibility layer
# ---------------------------------------------------------------------------

def _print_value(value):
    if isinstance(value, NativePointer):
        print(f"pointer:{value.address:#x}")
    elif isinstance(value, bytes):
        print(value.decode(errors="replace"))
    else:
        print(value)


def _cli():
    p = argparse.ArgumentParser(
        prog="ctypes.sh",
        description="Pure-CPython foreign-function interface compatible with "
                    "the core ctypes.sh operations.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    x = sub.add_parser("dlopen")
    x.add_argument("library")
    x.add_argument("flags", nargs="*")

    x = sub.add_parser("dlclose")
    x.add_argument("handle")

    x = sub.add_parser("dlsym")
    x.add_argument("symbol")
    x.add_argument("-h", "--handle")
    x.add_argument("-d", "--deref")

    x = sub.add_parser("dlcall")
    x.add_argument("symbol")
    x.add_argument("args", nargs="*")
    x.add_argument("-r", "--return-type", default="void")
    x.add_argument("-n", "--name", default="DLRETVAL")
    x.add_argument("-h", "--handle")

    x = sub.add_parser("sizeof")
    x.add_argument("type")

    x = sub.add_parser("unpack")
    x.add_argument("pointer")
    x.add_argument("types", nargs="+")

    x = sub.add_parser("alloc")
    x.add_argument("size", type=int)

    ns = p.parse_args()
    rt = CTypesRuntime()

    if ns.command == "dlopen":
        h = rt.dlopen(ns.library, *ns.flags)
        _print_value(h)
        return

    if ns.command == "dlclose":
        rt.dlclose(_parse_pointer(ns.handle))
        return

    if ns.command == "dlsym":
        h = _parse_pointer(ns.handle) if ns.handle else None
        sym = rt.dlsym(ns.symbol, h)
        if ns.deref:
            ctype = _ctype_for(ns.deref)
            obj = ctype.from_address(ctypes.cast(sym, ctypes.c_void_p).value)
            print(_format_scalar(ns.deref, obj.value))
        else:
            print(f"pointer:{ctypes.cast(sym, ctypes.c_void_p).value:#x}")
        return

    if ns.command == "dlcall":
        h = _parse_pointer(ns.handle) if ns.handle else None
        # Preserve ctypes.sh's untyped argument behavior.
        result = rt.dlcall(
            ns.symbol, *ns.args, restype=ns.return_type,
            handle=h, name=ns.name
        )
        if ns.return_type != "void":
            print(_format_scalar(ns.return_type, result))
        return

    if ns.command == "sizeof":
        print(rt.sizeof(ns.type))
        return

    if ns.command == "alloc":
        _print_value(alloc(ns.size))
        return

    if ns.command == "unpack":
        values = rt.unpack(_parse_pointer(ns.pointer), ns.types)
        print(" ".join(values))
        return


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

NULL = "pointer:0x0"
STDIN_FILENO = "int:0"
STDOUT_FILENO = "int:1"
STDERR_FILENO = "int:2"


if __name__ == "__main__":
    _cli()
