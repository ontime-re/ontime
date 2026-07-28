"""
Layout intermediate representation (IR) for onTime figures.

Every front-end (the :func:`ontime.rows` and :func:`ontime.cols` factories and,
later on, a string layout DSL) compiles down to the immutable tree defined
here :

- :class:`Panel` : a leaf, wrapping a single ``Plot``
- :class:`Rows` : vertical stacking of its children
- :class:`Cols` : horizontal placement of its children

Nodes are normalised **and interned** at construction time, which means that two
figures describing the same visual layout share the very same IR object. Layout
equality is therefore object identity, and ``repr`` is a stable, assertable,
symbolic rendering of the tree (no data access involved).
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence as AbcSequence
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

DEFAULT_SPACING = 4

# Interning table. Keys hold strong references to their children (and to the
# plot of a Panel), so a cached node keeps its own key alive ; once the node is
# garbage collected the entry disappears on its own.
_INTERNED: "weakref.WeakValueDictionary[tuple, LayoutNode]" = (
    weakref.WeakValueDictionary()
)

Size = Union[int, float, None]


class _Identity:
    """
    Hashable, identity based wrapper used to key a Panel by its plot object.

    Plots (and Altair charts) are not necessarily hashable nor comparable, hence
    this wrapper.
    """

    __slots__ = ("obj",)

    def __init__(self, obj: Any):
        self.obj = obj

    def __hash__(self) -> int:
        return id(self.obj)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, _Identity) and other.obj is self.obj


class LayoutNode:
    """
    Base class of the layout IR. Nodes are immutable and interned.
    """

    __slots__ = ("size", "__weakref__")

    #: symbolic operator used by ``__repr__``
    _operator: str = ""
    #: binding strength of the operator, the higher the tighter
    _precedence: int = 0

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable, build a new layout instead"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def with_size(self, size: Size) -> "LayoutNode":
        """
        Return an equivalent node carrying the given extent.

        :param size: extent along the stacking axis of the parent group, either
            an int (px) or a float (fraction of the parent extent)
        :return: LayoutNode
        """
        raise NotImplementedError

    def panels(self) -> Tuple["Panel", ...]:
        """
        Return the panels of the subtree, in reading order.

        :return: tuple of Panel
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return _render(self)


class Panel(LayoutNode):
    """
    Leaf of the layout IR, holding a single plot.

    :param plot: a ``Plot`` (or any Altair chart)
    :param size: extent along the stacking axis of the parent group
    """

    __slots__ = ("plot",)

    _precedence = 3

    def __new__(cls, plot: Any, size: Size = None) -> "Panel":
        key = (cls, _Identity(plot), size)
        node = _INTERNED.get(key)
        if node is None:
            node = super().__new__(cls)
            object.__setattr__(node, "plot", plot)
            object.__setattr__(node, "size", size)
            _INTERNED[key] = node
        return node  # type: ignore[return-value]

    def __init__(self, plot: Any, size: Size = None):
        # everything happens in __new__ since nodes are immutable and interned
        pass

    def with_size(self, size: Size) -> "Panel":
        """
        Return an equivalent panel carrying the given extent.

        :param size: extent along the stacking axis of the parent group
        :return: Panel
        """
        if size == self.size:
            return self
        return Panel(self.plot, size)

    def panels(self) -> Tuple["Panel", ...]:
        """
        Return the panel itself.

        :return: tuple of Panel
        """
        return (self,)


class Group(LayoutNode):
    """
    Base class of the composite nodes :class:`Rows` and :class:`Cols`.

    :param children: the child nodes
    :param share_x: whether the x scale domain is shared, ``None`` means
        "inherit from the enclosing group"
    :param share_y: whether the y scale domain is shared, ``None`` means
        "inherit from the enclosing group"
    :param spacing: inter-panel gap in px, ``None`` means "inherit"
    :param title: group title
    :param size: extent along the stacking axis of the parent group
    """

    __slots__ = ("children", "share_x", "share_y", "spacing", "title")

    #: axis along which the children are stacked, "y" for rows, "x" for cols
    _axis: str = ""

    def __new__(
        cls,
        children: Iterable[LayoutNode],
        share_x: Optional[bool] = None,
        share_y: Optional[bool] = None,
        spacing: Optional[int] = None,
        title: Optional[str] = None,
        size: Size = None,
    ) -> LayoutNode:
        children = tuple(children)
        if not children:
            raise ValueError(f"{cls.__name__} needs at least one panel, none was given")
        for child in children:
            if not isinstance(child, LayoutNode):
                raise TypeError(
                    f"{cls.__name__} children must be layout nodes, "
                    f"got {type(child).__name__}"
                )

        children = cls._flatten(children, share_x, share_y, spacing, title)

        # a group of one is its own child
        if len(children) == 1 and title is None:
            only = children[0]
            if isinstance(only, Panel) or (share_x is None and share_y is None):
                return only.with_size(size if size is not None else only.size)

        key = (cls, children, share_x, share_y, spacing, title, size)
        node = _INTERNED.get(key)
        if node is None:
            node = super().__new__(cls)
            object.__setattr__(node, "children", children)
            object.__setattr__(node, "share_x", share_x)
            object.__setattr__(node, "share_y", share_y)
            object.__setattr__(node, "spacing", spacing)
            object.__setattr__(node, "title", title)
            object.__setattr__(node, "size", size)
            _INTERNED[key] = node
        return node

    def __init__(
        self,
        children: Iterable[LayoutNode],
        share_x: Optional[bool] = None,
        share_y: Optional[bool] = None,
        spacing: Optional[int] = None,
        title: Optional[str] = None,
        size: Size = None,
    ):
        # everything happens in __new__ since nodes are immutable and interned
        pass

    @classmethod
    def _flatten(
        cls,
        children: Tuple[LayoutNode, ...],
        share_x: Optional[bool],
        share_y: Optional[bool],
        spacing: Optional[int],
        title: Optional[str],
    ) -> Tuple[LayoutNode, ...]:
        """
        Flatten same-orientation nesting, e.g. ``rows(rows(a, b), c)`` becomes
        ``rows(a, b, c)``.

        A child is only absorbed when it cannot carry any meaning of its own,
        i.e. it has no size, no title and no explicit sharing or spacing.

        :param children: the child nodes
        :param share_x: the group x sharing flag
        :param share_y: the group y sharing flag
        :param spacing: the group spacing
        :param title: the group title
        :return: tuple of LayoutNode
        """
        flat: list = []
        for child in children:
            absorbable = (
                type(child) is cls
                and child.size is None
                and child.title is None
                and child.share_x is None
                and child.share_y is None
                and child.spacing in (None, spacing)
            )
            if absorbable:
                flat.extend(child.children)
            else:
                flat.append(child)
        return tuple(flat)

    def with_size(self, size: Size) -> LayoutNode:
        """
        Return an equivalent group carrying the given extent.

        :param size: extent along the stacking axis of the parent group
        :return: LayoutNode
        """
        if size == self.size:
            return self
        return type(self)(
            self.children,
            share_x=self.share_x,
            share_y=self.share_y,
            spacing=self.spacing,
            title=self.title,
            size=size,
        )

    def panels(self) -> Tuple[Panel, ...]:
        """
        Return the panels of the subtree, in reading order.

        :return: tuple of Panel
        """
        found: list = []
        for child in self.children:
            found.extend(child.panels())
        return tuple(found)


class Rows(Group):
    """
    Vertical stacking of the children, i.e. each child is a row.
    """

    __slots__ = ()

    _axis = "y"
    _operator = " / "
    _precedence = 2


class Cols(Group):
    """
    Horizontal placement of the children, i.e. each child is a column.
    """

    __slots__ = ()

    _axis = "x"
    _operator = " | "
    _precedence = 1


def panel_name(plot: Any) -> Optional[str]:
    """
    Best effort symbolic name of a plot, read from its title.

    Only metadata is inspected, no data is ever touched.

    :param plot: a ``Plot`` or an Altair chart
    :return: str or None
    """
    title = getattr(plot, "title", None)
    if isinstance(title, str) and title:
        return title
    # a Plot keeps the title given to `.properties(title=...)`
    title = getattr(plot, "_title", None)
    if isinstance(title, str) and title:
        return title
    for layer in getattr(plot, "layers", []) or []:
        title = getattr(layer, "title", None)
        if isinstance(title, str) and title:
            return title
    return None


def _size_repr(size: Size) -> str:
    """
    Render a size annotation.

    :param size: the size to render
    :return: str
    """
    if size is None:
        return ""
    return f":{size}"


def _render(node: LayoutNode, names: Optional[dict] = None) -> str:
    """
    Render a layout tree symbolically, e.g. ``"a / b | c"``.

    :param node: the node to render
    :param names: internal cache of panel names
    :return: str
    """
    if names is None:
        names = {}
        for index, panel in enumerate(node.panels()):
            names[id(panel)] = panel_name(panel.plot) or f"p{index}"

    if isinstance(node, Panel):
        return f"{names[id(node)]}{_size_repr(node.size)}"

    parts = []
    for child in node.children:
        text = _render(child, names)
        if child._precedence < node._precedence:
            text = f"({text})"
        parts.append(text)
    rendered = node._operator.join(parts)
    if node.title is not None:
        rendered = f"{rendered} #{node.title}"
    if node.size is not None:
        rendered = f"({rendered}){_size_repr(node.size)}"
    return rendered


def normalise_sizes(
    sizes: Optional[Sequence[Union[int, float]]], count: int, axis: str
) -> Optional[Tuple[Union[int, float], ...]]:
    """
    Validate and normalise a list of per-child extents.

    Sizes are either a list of ints, understood as pixels, or a list of floats
    summing to 1.0, understood as fractions of the extent of the figure along
    the stacking axis. Mixing both is rejected.

    :param sizes: the sizes to validate
    :param count: the number of children of the group
    :param axis: "y" for rows (heights) or "x" for cols (widths)
    :return: tuple of int or float, or None
    """
    if sizes is None:
        return None

    extent = "heights" if axis == "y" else "widths"

    if isinstance(sizes, (str, bytes)) or not isinstance(sizes, AbcSequence):
        raise TypeError(f"sizes must be a list of {extent}, got {type(sizes).__name__}")

    sizes = tuple(sizes)
    if len(sizes) != count:
        raise ValueError(
            f"sizes has {len(sizes)} entries but the group has {count} panels, "
            f"give exactly one size per panel"
        )

    pixels = all(isinstance(s, int) and not isinstance(s, bool) for s in sizes)
    fractions = all(isinstance(s, float) for s in sizes)

    if not pixels and not fractions:
        raise ValueError(
            f"sizes must be either all ints (pixel {extent}, e.g. [240, 40]) or "
            f"all floats summing to 1.0 (fractions, e.g. [0.72, 0.28]), "
            f"got {list(sizes)}"
        )
    if any(s <= 0 for s in sizes):
        raise ValueError(f"sizes must be strictly positive, got {list(sizes)}")
    if fractions:
        total = sum(sizes)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"fractional sizes must sum to 1.0, got {list(sizes)} "
                f"summing to {total:g}"
            )
    return sizes
