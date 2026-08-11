"""
The layout string of onTime, a picture of a figure.

A layout string is a rectangular grid of characters, one character per cell :

    A A B
    C C B

Every character is a panel, ``.`` is a gap. A character repeated over adjacent
cells occupies the whole rectangle it covers, which expresses both spans and
proportions : ``A A B`` is a 2:1 horizontal split. The string describes
**geometry only**, there is no size, no scale linkage and no operator in it.

The grid is compiled to the layout IR of :mod:`._layout` by recursive bisection,
as in ``matplotlib.subplot_mosaic`` : full-width horizontal cuts give a
:class:`~._layout.Rows`, full-height vertical cuts give a
:class:`~._layout.Cols`, and a region holding a single repeated character gives a
:class:`~._layout.Panel` or a :class:`~._layout.Spacer`. A region with
interlocking spans, e.g. ``"A A B\\nC B B"``, is not expressible as nested rows
and columns, and Vega-Lite ``concat`` cannot render it either, hence it is
rejected at parse time.

:func:`to_string` is the inverse, it prints the minimal grid reproducing a layout
tree. Sizes and scale sharing groups are not part of the string.
"""

from __future__ import annotations

from math import gcd
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ._layout import (
    DEFAULT_SPACING,
    Cols,
    LayoutNode,
    Panel,
    Px,
    Rows,
    Size,
    Spacer,
)

#: the gap character, a cell that renders nothing
GAP = "."

#: characters a panel can be named after, in the order used by :func:`to_string`
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789"

Track = Union[int, float]
Cells = Tuple[Tuple[str, ...], ...]


# ------------------------------------------------------------------------ parsing


def parse(
    spec: str,
    widths: Optional[Sequence[Track]] = None,
    heights: Optional[Sequence[Track]] = None,
    spacing: int = DEFAULT_SPACING,
) -> LayoutNode:
    """
    Compile a layout string to a layout tree.

    The leaves of the returned tree hold the grid characters themselves, binding
    them to plots is the job of :func:`bind`.

    :param spec: the layout string, e.g. ``"A A B\\nC C B"``
    :param widths: one extent per grid column, ints (px) or floats (weights)
    :param heights: one extent per grid row, ints (px) or floats (weights)
    :param spacing: the inter-panel gap in px, used to derive the extent of a
        panel spanning several pixel-sized tracks
    :return: LayoutNode
    :raises SyntaxError: when the grid is empty, ragged or holds an unknown
        character
    :raises ValueError: when a character does not occupy a rectangle or when the
        layout has interlocking spans
    """
    cells = _lex(spec)
    _check_rectangles(cells)
    return _Bisector(cells, widths, heights, spacing).run()


def characters(cells_or_spec: Union[str, Cells]) -> Tuple[str, ...]:
    """
    Return the panel characters of a layout string, in reading order.

    :param cells_or_spec: a layout string or an already lexed grid
    :return: tuple of str
    """
    cells = _lex(cells_or_spec) if isinstance(cells_or_spec, str) else cells_or_spec
    seen: List[str] = []
    for row in cells:
        for char in row:
            if char != GAP and char not in seen:
                seen.append(char)
    return tuple(seen)


def shape(spec: str) -> Tuple[int, int]:
    """
    Return the number of rows and columns of a layout string.

    :param spec: the layout string
    :return: tuple of the row count and the column count
    """
    cells = _lex(spec)
    return len(cells), len(cells[0])


def bind(
    node: LayoutNode, panels: Dict[str, LayoutNode]
) -> Tuple[LayoutNode, Dict[str, Tuple[Panel, ...]]]:
    """
    Replace the grid characters of a parsed tree by the layouts they name.

    A character may name a whole nested figure, in which case its subtree is
    spliced in place of the leaf and the character maps to every panel of it.

    :param node: a tree returned by :func:`parse`
    :param panels: a mapping of grid character to layout node
    :return: tuple of the bound tree and of a mapping of character to the panels
        it names
    """
    bound: Dict[str, Tuple[Panel, ...]] = {}

    def walk(current: LayoutNode) -> LayoutNode:
        if isinstance(current, Panel):
            replacement = panels[current.plot].with_size(current.size)
            bound[current.plot] = replacement.panels()
            return replacement
        if isinstance(current, Spacer):
            return current
        return current.with_children([walk(child) for child in current.children])

    return walk(node), bound


def _lex(spec: str) -> Cells:
    """
    Turn a layout string into a rectangular grid of characters.

    All whitespace is meaningless and stripped, so ``"AAB\\nCCB"`` and its spaced,
    indented form are the same grid.

    :param spec: the layout string
    :return: tuple of tuple of str
    :raises SyntaxError: when the grid is empty, ragged or holds an unknown
        character
    """
    if not isinstance(spec, str):
        raise TypeError(f"a layout is described by a string, got {type(spec).__name__}")

    lines = [(number, line) for number, line in enumerate(spec.splitlines(), 1)]
    lines = [(number, line) for number, line in lines if line.strip()]
    if not lines:
        raise SyntaxError(
            "the layout string is empty, describe the figure with one character "
            'per cell, e.g. "A A B\\nC C B"'
        )

    rows: List[Tuple[str, ...]] = []
    width: Optional[int] = None
    for number, line in lines:
        cells = tuple("".join(line.split()))
        for char in cells:
            if char != GAP and not (char.isascii() and char.isalnum()):
                raise SyntaxError(
                    f"line {number} holds the unexpected character {char!r} : "
                    f"{line.strip()!r}, a cell is an ASCII letter, a digit or "
                    f"'.' for a gap, and there is no operator or bracket syntax"
                )
        if width is None:
            width = len(cells)
        elif len(cells) != width:
            raise SyntaxError(
                f"line {number} has {len(cells)} cells but {width} were expected "
                f": {line.strip()!r}, every line of a layout must hold the same "
                f"number of cells"
            )
        rows.append(cells)
    return tuple(rows)


def _check_rectangles(cells: Cells) -> None:
    """
    Check that every character occupies a solid rectangle.

    :param cells: the lexed grid
    :return: None
    :raises ValueError: when a character does not occupy a rectangle
    """
    positions: Dict[str, List[Tuple[int, int]]] = {}
    for row, line in enumerate(cells):
        for col, char in enumerate(line):
            if char != GAP:
                positions.setdefault(char, []).append((row, col))

    for char, found in positions.items():
        top = min(row for row, _ in found)
        bottom = max(row for row, _ in found)
        left = min(col for _, col in found)
        right = max(col for _, col in found)
        expected = (bottom - top + 1) * (right - left + 1)
        if len(found) != expected:
            raise ValueError(
                f"the character {char!r} does not occupy a rectangle, it covers "
                f"{len(found)} cells inside the rows {top}-{bottom} and the "
                f"columns {left}-{right}, which hold {expected} cells, a panel "
                f"must be a solid rectangle"
            )


class _Bisector:
    """
    Recursive bisection of a lexed grid into a layout tree.

    :param cells: the lexed grid
    :param widths: one extent per grid column, or None for uniform columns
    :param heights: one extent per grid row, or None for uniform rows
    :param spacing: the inter-panel gap in px
    """

    def __init__(
        self,
        cells: Cells,
        widths: Optional[Sequence[Track]],
        heights: Optional[Sequence[Track]],
        spacing: int,
    ):
        self.cells = cells
        self.widths = tuple(widths) if widths is not None else None
        self.heights = tuple(heights) if heights is not None else None
        self.spacing = spacing

    def run(self) -> LayoutNode:
        """
        Compile the whole grid.

        :return: LayoutNode
        """
        return self._region(0, len(self.cells), 0, len(self.cells[0]))

    def _region(self, top: int, bottom: int, left: int, right: int) -> LayoutNode:
        """
        Compile a rectangular region of the grid, bounds being half open.

        :param top: first row of the region
        :param bottom: row after the last one
        :param left: first column of the region
        :param right: column after the last one
        :return: LayoutNode
        """
        cuts = self._horizontal_cuts(top, bottom, left, right)
        if cuts:
            children = [
                self._region(start, stop, left, right).with_size(
                    self._size(self.heights, start, stop)
                )
                for start, stop in _bands(top, bottom, cuts)
            ]
            return Rows(children)

        cuts = self._vertical_cuts(top, bottom, left, right)
        if cuts:
            children = [
                self._region(top, bottom, start, stop).with_size(
                    self._size(self.widths, start, stop)
                )
                for start, stop in _bands(left, right, cuts)
            ]
            return Cols(children)

        found = {
            self.cells[row][col]
            for row in range(top, bottom)
            for col in range(left, right)
        } - {GAP}
        if not found:
            return Spacer()
        if len(found) == 1:
            return Panel(found.pop())
        raise ValueError(
            f"the region of the rows {top}-{bottom - 1} and the columns "
            f"{left}-{right - 1}, holding {''.join(sorted(found))}, has "
            f"interlocking spans and cannot be split into rows and columns, "
            f"build it with on.rows and on.cols instead"
        )

    def _horizontal_cuts(
        self, top: int, bottom: int, left: int, right: int
    ) -> List[int]:
        """
        Return every full-width horizontal cut of a region.

        :param top: first row of the region
        :param bottom: row after the last one
        :param left: first column of the region
        :param right: column after the last one
        :return: list of row indices a cut sits above
        """
        return [
            row
            for row in range(top + 1, bottom)
            if not any(
                self.cells[row][col] != GAP
                and self.cells[row][col] == self.cells[row - 1][col]
                for col in range(left, right)
            )
        ]

    def _vertical_cuts(self, top: int, bottom: int, left: int, right: int) -> List[int]:
        """
        Return every full-height vertical cut of a region.

        :param top: first row of the region
        :param bottom: row after the last one
        :param left: first column of the region
        :param right: column after the last one
        :return: list of column indices a cut sits left of
        """
        return [
            col
            for col in range(left + 1, right)
            if not any(
                self.cells[row][col] != GAP
                and self.cells[row][col] == self.cells[row][col - 1]
                for row in range(top, bottom)
            )
        ]

    def _size(self, tracks: Optional[Tuple[Track, ...]], start: int, stop: int) -> Size:
        """
        Derive the extent of a child from the grid tracks it spans.

        Without a track vector, the extent is the number of tracks spanned, i.e.
        the repetition of the character, taken as a relative weight. With pixel
        tracks, the extent is their sum plus the gaps between them.

        :param tracks: the track vector of the axis, or None for uniform tracks
        :param start: first track spanned
        :param stop: track after the last one
        :return: Size
        """
        count = stop - start
        if tracks is None:
            return count
        spanned = tracks[start:stop]
        if all(isinstance(track, int) for track in spanned):
            return Px(sum(spanned) + self.spacing * (count - 1))
        return float(sum(spanned))


def _bands(start: int, stop: int, cuts: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Split a range at the given cuts.

    :param start: first index
    :param stop: index after the last one
    :param cuts: the indices a cut sits before
    :return: list of half open ranges
    """
    edges = [start, *cuts, stop]
    return [(edges[index], edges[index + 1]) for index in range(len(edges) - 1)]


# ----------------------------------------------------------------------- printing


class _Token:
    """
    A single leaf occurrence of a layout tree, used while printing.

    Leaves are interned, so the same panel used twice would otherwise be printed
    as a single spanning panel.

    :param gap: whether the leaf is a spacer
    :param name: the grid character the leaf was parsed from, if any
    """

    __slots__ = ("gap", "name")

    def __init__(self, gap: bool, name: Optional[str] = None):
        self.gap = gap
        self.name = name


def to_string(node: LayoutNode) -> str:
    """
    Print the minimal layout string reproducing a layout tree.

    The grid is the smallest one honouring the proportions carried by the
    weights of the tree, using the least common multiple of the extents of the
    children along each axis. Sizes and scale sharing groups are **not** part of
    the string and never appear in the output.

    :param node: the layout tree to print
    :return: str
    :raises ValueError: when the layout holds more panels than there are
        characters available
    """
    grid = _paint(node)
    tokens = [token for row in grid for token in row if not token.gap]
    parsed = {token.name for token in tokens}
    # a tree straight out of `parse` knows the characters it came from, keeping
    # them is what makes the printer idempotent on a canonical layout string
    keep_names = None not in parsed and len(parsed) == len({id(t) for t in tokens})

    names: Dict[int, str] = {}
    for token in tokens:
        if id(token) in names:
            continue
        if keep_names:
            names[id(token)] = token.name
        elif len(names) >= len(ALPHABET):
            raise ValueError(
                f"a layout string can name at most {len(ALPHABET)} panels, "
                f"this figure holds more"
            )
        else:
            names[id(token)] = ALPHABET[len(names)]
    return "\n".join(
        " ".join(GAP if token.gap else names[id(token)] for token in row)
        for row in grid
    )


def _paint(node: LayoutNode) -> List[List[_Token]]:
    """
    Draw a layout tree on the smallest grid of leaf occurrences.

    :param node: the layout tree
    :return: list of rows of tokens
    """
    if isinstance(node, Spacer):
        return [[_Token(True)]]
    if isinstance(node, Panel):
        return [[_Token(False, _parsed_name(node))]]

    grids = [_paint(child) for child in node.children]
    weights = [_weight(child) for child in node.children]
    if isinstance(node, Cols):
        return _transpose(_stack([_transpose(grid) for grid in grids], weights))
    return _stack(grids, weights)


def _stack(grids: List[List[List[_Token]]], weights: List[int]) -> List[List[_Token]]:
    """
    Stack grids vertically, honouring their relative weights.

    Each grid is scaled up to a whole number of its own rows, so that the row
    counts follow the weights exactly.

    :param grids: the grids to stack
    :param weights: the relative weight of every grid
    :return: list of rows of tokens
    """
    factor = 1
    for grid, weight in zip(grids, weights):
        rows = len(grid)
        factor = _lcm(factor, rows // gcd(rows, weight))
    columns = 1
    for grid in grids:
        columns = _lcm(columns, len(grid[0]))

    stacked: List[List[_Token]] = []
    for grid, weight in zip(grids, weights):
        stacked.extend(_scale(grid, weight * factor, columns))
    return stacked


def _scale(grid: List[List[_Token]], rows: int, columns: int) -> List[List[_Token]]:
    """
    Replicate the cells of a grid up to the given shape.

    :param grid: the grid to scale
    :param rows: the target number of rows, a multiple of the current one
    :param columns: the target number of columns, a multiple of the current one
    :return: list of rows of tokens
    """
    vertical = rows // len(grid)
    horizontal = columns // len(grid[0])
    scaled: List[List[_Token]] = []
    for row in grid:
        wide = [token for token in row for _ in range(horizontal)]
        scaled.extend([list(wide) for _ in range(vertical)])
    return scaled


def _transpose(grid: List[List[_Token]]) -> List[List[_Token]]:
    """
    Transpose a grid.

    :param grid: the grid to transpose
    :return: list of rows of tokens
    """
    return [list(row) for row in zip(*grid)]


def _parsed_name(panel: Panel) -> Optional[str]:
    """
    Return the grid character a panel was parsed from, if it still holds one.

    :param panel: the panel to name
    :return: str or None
    """
    plot = panel.plot
    if isinstance(plot, str) and len(plot) == 1 and plot.isascii() and plot.isalnum():
        return plot
    return None


def _weight(node: LayoutNode) -> int:
    """
    Return the printable weight of a node, i.e. its repetition in the grid.

    Only integer weights are part of the geometry, pixel extents and fractional
    weights are sizes and therefore do not show up in a layout string.

    :param node: the node to measure
    :return: int
    """
    size = node.size
    if isinstance(size, int) and not isinstance(size, (bool, Px)) and size > 0:
        return int(size)
    return 1


def _lcm(left: int, right: int) -> int:
    """
    Least common multiple of two positive integers.

    :param left: the first integer
    :param right: the second integer
    :return: int
    """
    return left * right // gcd(left, right)
