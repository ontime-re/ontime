"""
Layout intermediate representation (IR) for onTime figures.

Every front-end (the :func:`ontime.layout` string parser and the
:func:`ontime.rows`, :func:`ontime.cols` and :func:`ontime.grid` factories)
compiles down to the immutable tree defined here :

- :class:`Panel` : a leaf, wrapping a single ``Plot``
- :class:`Spacer` : a leaf that occupies its slot and renders nothing
- :class:`Rows` : vertical stacking of its children
- :class:`Cols` : horizontal placement of its children

A node carries a ``size``, its extent along the stacking axis of its **parent**.
Two kinds of extent are distinguished :

- a relative weight, an ``int`` or a ``float``, normalised against the weights of
  its siblings, which is what the repetition of a character in a layout string
  produces
- an absolute number of pixels, wrapped in :class:`Px`, which is what integer
  ``widths`` and ``heights`` track vectors produce

Nodes are normalised **and interned** at construction time, which means that two
figures describing the same visual layout share the very same IR object. Layout
equality is therefore object identity, and ``repr`` is a stable, assertable
rendering of the layout as its canonical layout string (no data access involved).
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


class Px(int):
    """
    An extent given as an absolute number of pixels.

    A plain ``int`` or ``float`` size is a relative weight, whereas a ``Px`` size
    is absolute and therefore does not need the total extent of the figure to be
    known.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"Px({int(self)})"


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


class ShareGroups:
    """
    Named scale sharing groups, i.e. sets of panels sharing a scale domain.

    Panels outside of every group keep an independent domain. Since the groups
    are not necessarily subtrees of the layout, they are not rendered with the
    Vega-Lite ``resolve`` mechanism but by pinning the union domain of the group
    on each of its members.

    :param groups: an iterable of iterables of :class:`Panel`
    """

    __slots__ = ("groups",)

    def __init__(self, groups: Iterable[Iterable["Panel"]]):
        self.groups = tuple(tuple(group) for group in groups)

    def panels(self) -> Tuple["Panel", ...]:
        """
        Return every panel taking part in a group.

        :return: tuple of Panel
        """
        found: list = []
        for group in self.groups:
            found.extend(group)
        return tuple(found)

    def __hash__(self) -> int:
        return hash((ShareGroups, self.groups))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ShareGroups) and other.groups == self.groups

    def __repr__(self) -> str:
        return f"ShareGroups({list(self.groups)!r})"


#: a sharing flag, either a boolean or explicit groups of panels
Sharing = Union[bool, ShareGroups, None]


class LayoutNode:
    """
    Base class of the layout IR. Nodes are immutable and interned.
    """

    __slots__ = ("size", "__weakref__")

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
            a relative weight (int or float) or a :class:`Px` number of pixels
        :return: LayoutNode
        """
        raise NotImplementedError

    def panels(self) -> Tuple["Panel", ...]:
        """
        Return the panels of the subtree, in reading order.

        :return: tuple of Panel
        """
        raise NotImplementedError

    def leaves(self) -> Tuple["LayoutNode", ...]:
        """
        Return the leaves of the subtree, panels and spacers, in reading order.

        :return: tuple of LayoutNode
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        from ._grid import to_string

        return to_string(self)


class Panel(LayoutNode):
    """
    Leaf of the layout IR, holding a single plot.

    :param plot: a ``Plot`` (or any Altair chart)
    :param size: extent along the stacking axis of the parent group
    """

    __slots__ = ("plot",)

    def __new__(cls, plot: Any, size: Size = None) -> "Panel":
        size = _normalise_size(size)
        key = (cls, _Identity(plot), type(size), size)
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
        if _same_size(size, self.size):
            return self
        return Panel(self.plot, size)

    def panels(self) -> Tuple["Panel", ...]:
        """
        Return the panel itself.

        :return: tuple of Panel
        """
        return (self,)

    def leaves(self) -> Tuple[LayoutNode, ...]:
        """
        Return the panel itself.

        :return: tuple of LayoutNode
        """
        return (self,)


class Spacer(LayoutNode):
    """
    Leaf of the layout IR occupying its slot and rendering nothing.

    A spacer is what a ``.`` cell of a layout string compiles to.

    :param size: extent along the stacking axis of the parent group
    """

    __slots__ = ()

    def __new__(cls, size: Size = None) -> "Spacer":
        size = _normalise_size(size)
        key = (cls, type(size), size)
        node = _INTERNED.get(key)
        if node is None:
            node = super().__new__(cls)
            object.__setattr__(node, "size", size)
            _INTERNED[key] = node
        return node  # type: ignore[return-value]

    def __init__(self, size: Size = None):
        # everything happens in __new__ since nodes are immutable and interned
        pass

    def with_size(self, size: Size) -> "Spacer":
        """
        Return an equivalent spacer carrying the given extent.

        :param size: extent along the stacking axis of the parent group
        :return: Spacer
        """
        if _same_size(size, self.size):
            return self
        return Spacer(size)

    def panels(self) -> Tuple[Panel, ...]:
        """
        Return no panel, a spacer holds no plot.

        :return: empty tuple
        """
        return ()

    def leaves(self) -> Tuple[LayoutNode, ...]:
        """
        Return the spacer itself.

        :return: tuple of LayoutNode
        """
        return (self,)


class Group(LayoutNode):
    """
    Base class of the composite nodes :class:`Rows` and :class:`Cols`.

    :param children: the child nodes
    :param size: extent along the stacking axis of the parent group
    :param share_x: whether the x scale domain is shared, either a boolean, a
        :class:`ShareGroups` or ``None``, which means "inherit from the enclosing
        group"
    :param share_y: whether the y scale domain is shared, same forms as
        ``share_x``
    :param spacing: inter-panel gap in px, ``None`` means "inherit"
    :param title: group title
    :param labels: ``"bottom"`` to draw inner x axis labels only on the bottom
        row, ``"all"`` to draw them on every panel, ``None`` means "inherit"
    """

    __slots__ = ("children", "share_x", "share_y", "spacing", "title", "labels")

    #: axis along which the children are stacked, "y" for rows, "x" for cols
    _axis: str = ""

    def __new__(
        cls,
        children: Iterable[LayoutNode],
        size: Size = None,
        *,
        share_x: Sharing = None,
        share_y: Sharing = None,
        spacing: Optional[int] = None,
        title: Optional[str] = None,
        labels: Optional[str] = None,
    ) -> LayoutNode:
        size = _normalise_size(size)
        children = tuple(children)
        if not children:
            raise ValueError(f"{cls.__name__} needs at least one panel, none was given")
        for child in children:
            if not isinstance(child, LayoutNode):
                raise TypeError(
                    f"{cls.__name__} children must be layout nodes, "
                    f"got {type(child).__name__}"
                )

        children = cls._flatten(children, share_x, share_y, spacing, title, labels)

        # a group of one is its own child
        if len(children) == 1 and title is None:
            only = children[0]
            if not isinstance(only, Group) or (
                share_x is None and share_y is None and labels is None
            ):
                return only.with_size(size if size is not None else only.size)

        key = (
            cls,
            children,
            share_x,
            share_y,
            spacing,
            title,
            labels,
            type(size),
            size,
        )
        node = _INTERNED.get(key)
        if node is None:
            node = super().__new__(cls)
            object.__setattr__(node, "children", children)
            object.__setattr__(node, "share_x", share_x)
            object.__setattr__(node, "share_y", share_y)
            object.__setattr__(node, "spacing", spacing)
            object.__setattr__(node, "title", title)
            object.__setattr__(node, "labels", labels)
            object.__setattr__(node, "size", size)
            _INTERNED[key] = node
        return node

    def __init__(
        self,
        children: Iterable[LayoutNode],
        size: Size = None,
        *,
        share_x: Sharing = None,
        share_y: Sharing = None,
        spacing: Optional[int] = None,
        title: Optional[str] = None,
        labels: Optional[str] = None,
    ):
        # everything happens in __new__ since nodes are immutable and interned
        pass

    @classmethod
    def _flatten(
        cls,
        children: Tuple[LayoutNode, ...],
        share_x: Sharing,
        share_y: Sharing,
        spacing: Optional[int],
        title: Optional[str],
        labels: Optional[str],
    ) -> Tuple[LayoutNode, ...]:
        """
        Flatten same-orientation nesting, e.g. ``rows(rows(a, b), c)`` becomes
        ``rows(a, b, c)``.

        A child is only absorbed when it cannot carry any meaning of its own,
        i.e. it has no size, no title, no labels policy and no explicit sharing
        or spacing. The weights of the absorbed grandchildren are kept as they
        are, an absorbed group having no weight of its own.

        :param children: the child nodes
        :param share_x: the group x sharing flag
        :param share_y: the group y sharing flag
        :param spacing: the group spacing
        :param title: the group title
        :param labels: the group labels policy
        :return: tuple of LayoutNode
        """
        flat: list = []
        for child in children:
            absorbable = (
                type(child) is cls
                and child.size is None
                and child.title is None
                and child.labels is None
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
        if _same_size(size, self.size):
            return self
        return type(self)(
            self.children,
            size,
            share_x=self.share_x,
            share_y=self.share_y,
            spacing=self.spacing,
            title=self.title,
            labels=self.labels,
        )

    def with_children(self, children: Iterable[LayoutNode]) -> LayoutNode:
        """
        Return an equivalent group built on the given children.

        :param children: the new child nodes
        :return: LayoutNode
        """
        return type(self)(
            children,
            self.size,
            share_x=self.share_x,
            share_y=self.share_y,
            spacing=self.spacing,
            title=self.title,
            labels=self.labels,
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

    def leaves(self) -> Tuple[LayoutNode, ...]:
        """
        Return the leaves of the subtree, panels and spacers, in reading order.

        :return: tuple of LayoutNode
        """
        found: list = []
        for child in self.children:
            found.extend(child.leaves())
        return tuple(found)


class Rows(Group):
    """
    Vertical stacking of the children, i.e. each child is a row.
    """

    __slots__ = ()

    _axis = "y"


class Cols(Group):
    """
    Horizontal placement of the children, i.e. each child is a column.
    """

    __slots__ = ()

    _axis = "x"


def _normalise_size(size: Size) -> Size:
    """
    Normalise an extent, the neutral weight ``1`` being the same as no extent.

    A group of equally weighted children is the default, so a weight of ``1``
    carries no information and is dropped, which keeps the interning table free
    of duplicates and lets ``rows(rows(a, b), c)`` still flatten. A ``Px(1)``
    extent is absolute and therefore kept.

    :param size: the extent to normalise
    :return: the extent or None
    """
    if isinstance(size, bool):
        raise TypeError(f"a size must be a number or Px, got {size!r}")
    if type(size) is int and size == 1:
        return None
    return size


def _same_size(left: Size, right: Size) -> bool:
    """
    Compare two extents, a weight never being equal to a pixel extent.

    :param left: the first extent
    :param right: the second extent
    :return: bool
    """
    left, right = _normalise_size(left), _normalise_size(right)
    return type(left) is type(right) and left == right


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


def normalise_tracks(
    tracks: Optional[Sequence[Union[int, float]]],
    count: int,
    axis: str,
) -> Optional[Tuple[Union[int, float], ...]]:
    """
    Validate a vector of grid track extents.

    A track vector holds one entry per grid row (``heights``) or per grid column
    (``widths``). Units must be homogeneous within a vector, ``int`` entries
    being pixels and ``float`` entries relative weights normalised by the sum of
    the vector. The two vectors are independent, so pixel heights next to
    fractional widths is legal.

    :param tracks: the track extents to validate
    :param count: the number of grid tracks along the axis
    :param axis: "y" for rows (heights) or "x" for cols (widths)
    :return: tuple of int or float, or None
    """
    if tracks is None:
        return None

    name = "heights" if axis == "y" else "widths"
    track = "row" if axis == "y" else "column"

    if isinstance(tracks, (str, bytes)) or not isinstance(tracks, AbcSequence):
        raise TypeError(
            f"{name} must be a list of extents, got {type(tracks).__name__}"
        )

    tracks = tuple(tracks)
    if len(tracks) != count:
        raise ValueError(
            f"{name} has {len(tracks)} entries but the grid has {count} "
            f"{track}s, give exactly one entry per grid {track}"
        )

    pixels = all(isinstance(t, int) and not isinstance(t, bool) for t in tracks)
    fractions = all(isinstance(t, float) for t in tracks)

    if not pixels and not fractions:
        raise ValueError(
            f"{name} must be either all ints (pixels, e.g. [240, 40]) or all "
            f"floats (relative weights, e.g. [0.3, 0.3, 0.4]), units cannot be "
            f"mixed within a vector, got {list(tracks)}"
        )
    if any(t <= 0 for t in tracks):
        raise ValueError(f"{name} must be strictly positive, got {list(tracks)}")
    return tracks
