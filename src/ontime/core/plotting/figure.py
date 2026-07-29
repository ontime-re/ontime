"""
Figure, the subplot container of onTime.

A ``Plot`` is a single panel, made of layered marks. A :class:`Figure` places
several panels next to each other. Figures are built with :func:`layout`, which
draws the arrangement as a string, or with the factories :func:`rows`,
:func:`cols` and :func:`grid`, and they nest freely.

    import ontime as on

    on.rows(
        on.Plot(solar).add(on.marks.line),
        on.Plot(nuclear).add(on.marks.line),
    ).properties(width=800, height=140).show()

    on.layout(
        '''
        A A B
        C C B
        ''',
        A=on.Plot(solar).add(on.marks.line),
        B=on.Plot(nuclear).add(on.marks.line),
        C=on.Plot(total).add(on.marks.line),
    ).properties(width=300, height=150).show()

Naming : ``rows(a, b)`` reads as "a and b are rows", i.e. they are stacked
vertically. The factories describe their arguments, not the container, which
avoids the usual ``vstack`` / ``hstack`` ambiguity.

Sizes are never part of a layout string. The ``width`` and ``height`` of a figure
are the extent of **one panel**, so a figure of three columns is about three times
as wide, and a panel spanning several tracks is as long as the tracks it covers,
gaps included.

Tracks are sized individually with the vectors ``heights=`` for the grid rows and
``widths=`` for the grid columns, one entry per track, all ints (px) or all floats
(relative weights). Weights are shares of the extent of the figure, which is then
read as the total of that axis rather than as the extent of one panel.

Scale sharing propagates : a ``share_x`` or ``share_y`` given **explicitly** to a
group is inherited by its nested groups, unless the nested call sets the flag
itself. A flag left unset nowhere in the chain falls back to the default of the
group kind, i.e. ``share_x=True`` and ``share_y=False`` for rows and layouts,
``share_x=False`` and ``share_y=False`` for cols, and both ``True`` for grids.
This is the most likely source of surprise when nesting figures.

Sharing is not limited to booleans : panels that are not a group of their own are
shared by naming them, ``share_y="AC"`` for one group of two panels, or
``share_y=["AC", "BD"]`` for two independent groups. Every string is one group, so
``["A", "C"]`` shares nothing. A named group is rendered by pinning the union of
the data domains of its panels on each of them.

Placement : panels are measured with their axes and titles (``bounds="full"``)
and separated by ``spacing`` px, so they never run over each other. A shared axis
is drawn only once, on the bottom row for x and on the leftmost column for y, and
``axis_extent`` px are reserved for every y axis so that the plotting areas stay
aligned. All of this is tunable through :meth:`Figure.properties`.

Known constraint : in Vega-Lite 5, ``selection_interval(bind="scales")`` does not
reliably propagate across concatenated views, therefore synchronised pan and zoom
across panels is not supported. Sharing a scale domain (``share_x`` /
``share_y``) works, interactive linking does not.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence as AbcSequence
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import altair as alt

from . import _grid
from ._layout import (
    DEFAULT_SPACING,
    Cols,
    Group,
    LayoutNode,
    Panel,
    Px,
    Rows,
    ShareGroups,
    Sharing,
    Spacer,
    normalise_tracks,
)

Panelish = Any  # Plot, Figure or Altair chart
Number = Union[int, float]

# Scale sharing defaults, applied when a flag can neither be read from the node
# nor inherited from an enclosing group.
_DEFAULT_SHARING = {
    Rows: {"x": True, "y": False},
    Cols: {"x": False, "y": False},
}

# Scale sharing defaults of the front-ends, recorded on the top node of the
# figure they build so that they also reach the groups nested under it. ``None``
# means "whatever that node defaults to", which keeps the axis open to the
# sharing of an enclosing group.
_KIND_SHARING = {
    "rows": {"x": None, "y": None},
    "cols": {"x": None, "y": None},
    "grid": {"x": True, "y": True},
    "layout": {"x": True, "y": False},
}

#: how inner x axis labels are drawn, ``"bottom"`` only on the bottom row of a
#: group sharing its x scale, ``"all"`` on every panel
DEFAULT_LABELS = "bottom"

#: extent of a grid track, in px, used when a figure has no explicit extent and
#: its tracks are not all of the same weight
DEFAULT_TRACK_WIDTH = 200
DEFAULT_TRACK_HEIGHT = 100

#: how the extent of a panel is measured when panels are concatenated,
#: ``"full"`` counts the axes and the titles, ``"flush"`` only the plotting area
DEFAULT_BOUNDS = "full"

#: axis properties of a hidden axis, the extents being zeroed so that the axis
#: leaves no room between the panels
_HIDDEN_AXIS = {
    "labels": False,
    "ticks": False,
    "title": None,
    "minExtent": 0,
    "maxExtent": 0,
}

#: minimum room reserved for the y axis of a panel, in px, so that the plotting
#: areas of stacked panels start at the same x position
DEFAULT_AXIS_EXTENT = 40


class _Extent:
    """
    Sizing context handed down the layout tree during compilation.

    :param width: total width available for the node
    :param height: total height available for the node
    :param panel_width: default width of a panel of the node
    :param panel_height: default height of a panel of the node
    """

    __slots__ = ("width", "height", "panel_width", "panel_height")

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        panel_width: Optional[int] = None,
        panel_height: Optional[int] = None,
    ):
        self.width = width
        self.height = height
        self.panel_width = panel_width
        self.panel_height = panel_height


class Figure:
    """
    A composition of panels, compiled to concatenated Altair views.

    Figures are normally created by :func:`rows` and :func:`cols`
    rather than instantiated directly.

    :param layout: the layout IR of the figure
    :param source: how the figure was built, kept so that ``widths`` and
        ``heights`` can still be given to :meth:`properties`
    """

    def __init__(self, layout: LayoutNode, source: Optional["_Source"] = None):
        if not isinstance(layout, LayoutNode):
            raise TypeError(
                f"Figure expects a layout node, got {type(layout).__name__}"
            )
        self._layout = layout
        self._source = source
        self._chars: Dict[str, Tuple[Panel, ...]] = {}
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._spacing: Optional[int] = None
        self._title: Optional[str] = None
        self._resolve: Optional[Dict[str, Dict[str, str]]] = None
        self._bounds: Optional[str] = None
        self._axis_extent: Optional[int] = None
        self._hide_shared_axes: Optional[bool] = None

    # ------------------------------------------------------------------ public

    @property
    def layout(self) -> LayoutNode:
        """
        The layout IR of the figure.

        :return: LayoutNode
        """
        return self._layout

    @property
    def title(self) -> Optional[str]:
        """
        The figure title, if any.

        :return: str or None
        """
        return self._title

    def to_string(self) -> str:
        """
        Return the canonical layout string of the figure.

        The string is the picture of the arrangement only, sizes and scale
        sharing groups are not part of it. Parsing it back gives the same
        geometry, and printing it again gives the very same string.

        :return: str
        """
        return _grid.to_string(self._layout)

    def properties(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        widths: Optional[Sequence[Number]] = None,
        heights: Optional[Sequence[Number]] = None,
        spacing: Optional[int] = None,
        title: Optional[str] = None,
        resolve: Optional[Dict[str, Dict[str, str]]] = None,
        bounds: Optional[str] = None,
        axis_extent: Optional[int] = None,
        hide_shared_axes: Optional[bool] = None,
    ) -> "Figure":
        """
        Set figure level properties.

        Figure level properties only fill values that panels left unset, panel
        level ``.properties()`` always wins. ``width`` and ``height`` are the
        default extents of a **single panel**, whereas ``widths`` and ``heights``
        are the extents of the grid **tracks** of the figure, one entry per grid
        column and per grid row. A panel spanning several tracks is as long as the
        tracks it covers, gaps included, so a figure is usually larger than
        ``width`` by ``height``. An axis whose tracks are given as relative weights
        is the exception : there ``width`` or ``height`` is the total to share.

        ``spacing`` is the gap left between panels. Since the default
        ``bounds="full"`` measures a panel with its axes and its title, the gap
        is the room between those, not between the plotting areas. Increase it
        when panels look crowded, and use ``bounds="flush"`` to measure the
        plotting areas only, which packs panels tightly but lets axes and titles
        run over the neighbouring panel.

        :param width: default panel width in px
        :param height: default panel height in px
        :param widths: one extent per grid column, all ints (px) or all floats
            (relative weights)
        :param heights: one extent per grid row, same units as ``widths``
        :param spacing: inter-panel gap in px, defaults to 4
        :param title: figure title
        :param resolve: raw Vega-Lite resolve dict, e.g.
            ``{"scale": {"y": "independent"}, "legend": {"color": "shared"}}``,
            applied last and therefore overriding the ``share_x`` /
            ``share_y`` flags
        :param bounds: ``"full"`` (default) to measure panels with their axes and
            titles, ``"flush"`` to measure their plotting areas only
        :param axis_extent: minimum room reserved for the y axis of a panel in
            px, defaults to 40, which keeps the plotting areas of stacked panels
            aligned even when their labels have different widths. Raise it for
            wide labels, set ``0`` to let every panel size its own axis.
        :param hide_shared_axes: whether a shared axis is drawn only once,
            defaults to ``True``. Set to ``False`` to keep the redundant axis on
            every panel.
        :return: Figure
        """
        if width is not None:
            self._width = width
        if height is not None:
            self._height = height
        if spacing is not None:
            self._spacing = spacing
        if title is not None:
            self._title = title
        if resolve is not None:
            if not isinstance(resolve, dict):
                raise TypeError(
                    f"resolve must be a Vega-Lite resolve dict, "
                    f"got {type(resolve).__name__}"
                )
            self._resolve = resolve
        if bounds is not None:
            if bounds not in ("full", "flush"):
                raise ValueError(f"bounds must be 'full' or 'flush', got {bounds!r}")
            self._bounds = bounds
        if axis_extent is not None:
            if (
                not isinstance(axis_extent, int)
                or isinstance(axis_extent, bool)
                or axis_extent < 0
            ):
                raise ValueError(
                    f"axis_extent must be a positive int in px, got {axis_extent!r}"
                )
            self._axis_extent = axis_extent
        if hide_shared_axes is not None:
            self._hide_shared_axes = bool(hide_shared_axes)
        if widths is not None or heights is not None:
            self._retrack(widths, heights)
        # panels of a shared x axis must be equally wide to stay aligned
        _Compiler(self, build=False).run()
        return self

    def _retrack(
        self,
        widths: Optional[Sequence[Number]],
        heights: Optional[Sequence[Number]],
    ) -> None:
        """
        Rebuild the layout on new track extents, in place.

        :param widths: one extent per grid column, or None to keep the current
            ones
        :param heights: one extent per grid row, or None to keep the current ones
        :return: None
        """
        source = self._source
        if source is None:
            if widths is None and heights is None:
                return
            raise ValueError(
                "widths and heights can only be set on a figure built by "
                "on.layout, on.rows, on.cols or on.grid"
            )
        source = source.replace(widths=widths, heights=heights)
        self._layout, self._chars = _assemble(source, self._spacing)
        self._source = source

    def to_altair(self) -> alt.TopLevelMixin:
        """
        Compile the figure to Altair, the escape hatch to plain Altair objects.

        :return: Altair chart
        """
        return _Compiler(self).run()

    def show(self) -> alt.TopLevelMixin:
        """
        Show the figure.

        :return: Altair chart
        """
        return self.to_altair()

    def save(self, path: str, **kwargs: Any) -> None:
        """
        Save the figure to a file, in html, png, svg or json format.

        :param path: destination path, the extension sets the format
        :param kwargs: additional arguments passed to Altair
        :return: None
        """
        supported = (".html", ".png", ".svg", ".json")
        if not str(path).lower().endswith(supported):
            raise ValueError(
                f"cannot save to '{path}', supported extensions are "
                f"{', '.join(supported)}"
            )
        self.to_altair().save(path, **kwargs)

    # ------------------------------------------------------------ presentation

    def __repr__(self) -> str:
        return self.to_string()

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """
        Render the figure in notebooks, without an explicit call to ``show()``.

        :param include: mime types to include
        :param exclude: mime types to exclude
        :return: mime bundle
        """
        return self.to_altair()._repr_mimebundle_(include, exclude)


class _Compiler:
    """
    Compile a layout tree to Altair, or only validate it.

    :param figure: the figure to compile
    :param build: whether Altair objects are built, ``False`` only runs the
        checks (used by ``Figure.properties``)
    """

    def __init__(self, figure: Figure, build: bool = True):
        self.figure = figure
        self.build = build
        self.bounds = figure._bounds if figure._bounds is not None else DEFAULT_BOUNDS
        self.hide_shared_axes = (
            True if figure._hide_shared_axes is None else figure._hide_shared_axes
        )
        self.axis_extent = (
            DEFAULT_AXIS_EXTENT if figure._axis_extent is None else figure._axis_extent
        )
        self._inspected: Dict[int, alt.TopLevelMixin] = {}
        self.domains: Dict[int, Dict[str, list]] = {}

    def run(self) -> Optional[alt.TopLevelMixin]:
        """
        Compile the whole figure.

        :return: Altair chart or None when only validating
        """
        figure = self.figure
        self.domains = self._share_group_domains(figure._layout)
        context = _Extent(
            width=figure._width,
            height=figure._height,
            panel_width=figure._width,
            panel_height=figure._height,
        )
        chart = self._node(
            figure._layout,
            context,
            share_x=None,
            share_y=None,
            hide_x=False,
            hide_y=False,
            align_width=False,
            labels=None,
        )
        if not self.build:
            return None

        if self.axis_extent and isinstance(figure._layout, Group):
            # reserve the same room for every y axis, so that the plotting areas
            # of the panels start at the same x position
            chart = chart.configure_axisY(minExtent=self.axis_extent)
        if figure._title is not None:
            chart = chart.properties(title=figure._title)
        if figure._resolve is not None:
            chart = _apply_raw_resolve(chart, figure._resolve)
        return chart

    # ------------------------------------------------------------------ nodes

    def _node(
        self,
        node: LayoutNode,
        context: _Extent,
        share_x: Sharing,
        share_y: Sharing,
        hide_x: bool,
        hide_y: bool,
        align_width: bool,
        labels: Optional[str],
    ) -> Optional[alt.TopLevelMixin]:
        """
        Compile a single node of the layout tree.

        :param node: the node to compile
        :param context: the sizing context of the node
        :param share_x: x sharing explicitly set by an enclosing group, or None
        :param share_y: y sharing explicitly set by an enclosing group, or None
        :param hide_x: whether the x axis of the node must be hidden
        :param hide_y: whether the y axis of the node must be hidden
        :param align_width: whether the node is stacked under a shared x axis
        :param labels: the labels policy of the enclosing group, or None
        :return: Altair chart or None when only validating
        """
        if isinstance(node, Spacer):
            return self._spacer(context)
        if isinstance(node, Panel):
            return self._panel(node, context, hide_x, hide_y, align_width)
        return self._group(
            node, context, share_x, share_y, hide_x, hide_y, align_width, labels
        )

    def _spacer(self, context: _Extent) -> Optional[alt.TopLevelMixin]:
        """
        Compile a spacer, an empty view holding its slot.

        :param context: the sizing context of the spacer
        :return: Altair chart or None when only validating
        """
        if not self.build:
            return None
        chart = alt.Chart(alt.Data(values=[{}])).mark_point(opacity=0)
        dimensions = {}
        if context.panel_width is not None:
            dimensions["width"] = context.panel_width
        if context.panel_height is not None:
            dimensions["height"] = context.panel_height
        return chart.properties(**dimensions) if dimensions else chart

    def _panel(
        self,
        node: Panel,
        context: _Extent,
        hide_x: bool,
        hide_y: bool,
        align_width: bool,
    ) -> Optional[alt.TopLevelMixin]:
        """
        Compile a leaf of the layout tree.

        :param node: the panel to compile
        :param context: the sizing context of the panel
        :param hide_x: whether the x axis of the panel must be hidden
        :param hide_y: whether the y axis of the panel must be hidden
        :param align_width: whether the panel is stacked under a shared x axis
        :return: Altair chart or None when only validating
        """
        chart = _panel_chart(node.plot)
        set_width, set_height = _explicit_dims(chart)

        if (
            align_width
            and set_width is not None
            and context.panel_width is not None
            and set_width != context.panel_width
        ):
            raise ValueError(
                f"panel width {set_width} conflicts with the figure width "
                f"{context.panel_width}, stacked panels sharing an x axis must "
                f"be equally wide, either drop the panel width or set the same "
                f"width on the figure"
            )

        if not self.build:
            return None

        # panel level properties win, figure level ones only fill the gaps
        dimensions = {}
        if set_width is None and context.panel_width is not None:
            dimensions["width"] = context.panel_width
        if set_height is None and context.panel_height is not None:
            dimensions["height"] = context.panel_height
        if dimensions:
            chart = chart.properties(**dimensions)
        for channel, domain in self.domains.get(id(node), {}).items():
            _pin_domain(chart, channel, domain)
        if hide_x:
            _hide_axis(chart, "x")
        if hide_y:
            _hide_axis(chart, "y")
        return chart

    def _group(
        self,
        node: Group,
        context: _Extent,
        share_x: Sharing,
        share_y: Sharing,
        hide_x: bool,
        hide_y: bool,
        align_width: bool,
        labels: Optional[str],
    ) -> Optional[alt.TopLevelMixin]:
        """
        Compile a group of the layout tree.

        A shared axis is drawn only once, on the bottom row for a shared x and on
        the leftmost column for a shared y, unless ``labels="all"`` or
        ``hide_shared_axes=False`` was given.

        Only the boolean forms of ``share_x`` / ``share_y`` are rendered with the
        Vega-Lite ``resolve`` mechanism, named groups pin their union domain on
        each of their members instead, see :meth:`_share_group_domains`.

        :param node: the group to compile
        :param context: the sizing context of the group
        :param share_x: x sharing explicitly set by an enclosing group, or None
        :param share_y: y sharing explicitly set by an enclosing group, or None
        :param hide_x: whether the x axis of the group must be hidden
        :param hide_y: whether the y axis of the group must be hidden
        :param align_width: whether the group is stacked under a shared x axis
        :param labels: the labels policy of the enclosing group, or None
        :return: Altair chart or None when only validating
        """
        # an explicit flag wins, then any flag explicitly set upstream, then the
        # default of the group kind
        default = _DEFAULT_SHARING[type(node)]
        explicit_x = node.share_x if node.share_x is not None else share_x
        explicit_y = node.share_y if node.share_y is not None else share_y
        shared_x = _resolve_flag(explicit_x, default["x"]) is True
        shared_y = _resolve_flag(explicit_y, default["y"]) is True
        group_labels = node.labels if node.labels is not None else labels
        keep_labels = (group_labels or DEFAULT_LABELS) == "all"
        spacing = _resolve_spacing(node.spacing, self.figure._spacing)

        vertical = isinstance(node, Rows)
        extents = self._extents(node, context, spacing, vertical)
        cross = self._cross_extent(node, context, vertical)
        children_align = align_width or (vertical and shared_x)

        if vertical and shared_x:
            self._check_equal_widths(node, context)

        charts: List[alt.TopLevelMixin] = []
        last = len(node.children) - 1
        for index, child in enumerate(node.children):
            child_hide_x = hide_x or (
                self.hide_shared_axes
                and not keep_labels
                and vertical
                and shared_x
                and index != last
            )
            child_hide_y = hide_y or (
                self.hide_shared_axes and not vertical and shared_y and index != 0
            )
            chart = self._node(
                child,
                self._child_context(
                    child, context, extents[index], cross, spacing, vertical
                ),
                share_x=explicit_x,
                share_y=explicit_y,
                hide_x=child_hide_x,
                hide_y=child_hide_y,
                align_width=children_align,
                labels=group_labels,
            )
            charts.append(chart)

        if not self.build:
            return None

        concatenate = alt.vconcat if vertical else alt.hconcat
        chart = _concatenate(concatenate, charts, spacing, self.bounds)
        chart = chart.resolve_scale(
            x="shared" if shared_x else "independent",
            y="shared" if shared_y else "independent",
        )
        if node.title is not None:
            chart = chart.properties(title=node.title)
        return chart

    # ----------------------------------------------------------------- sharing

    def _share_group_domains(self, layout: LayoutNode) -> Dict[int, Dict[str, list]]:
        """
        Resolve the domain every panel of a named sharing group must be pinned to.

        Named groups are not necessarily subtrees of the layout, so Vega-Lite
        ``resolve`` cannot express them. The union domain of a group is computed
        from the data of its panels and pinned on each of them, which is what
        makes their scales identical.

        :param layout: the layout tree of the figure
        :return: mapping of panel id to a mapping of channel to domain
        """
        assignments: Dict[int, Dict[str, list]] = {}
        if not self.build:
            return assignments
        for node in _walk(layout):
            if not isinstance(node, Group):
                continue
            for channel, share in (("x", node.share_x), ("y", node.share_y)):
                if not isinstance(share, ShareGroups):
                    continue
                for group in share.groups:
                    domain = self._union_domain(group, channel)
                    if domain is None:
                        continue
                    for panel in group:
                        assignments.setdefault(id(panel), {})[channel] = domain
        return assignments

    def _union_domain(
        self, panels: Sequence[Panel], channel: str
    ) -> Optional[List[Any]]:
        """
        Compute the union domain of a group of panels along a channel.

        :param panels: the panels of the group
        :param channel: ``"x"`` or ``"y"``
        :return: a two element domain, or None when it cannot be read
        """
        lows: List[Any] = []
        highs: List[Any] = []
        for panel in panels:
            found = _channel_domain(self._inspect(panel.plot), channel)
            if found is None:
                continue
            lows.append(found[0])
            highs.append(found[1])
        if not lows:
            return None
        try:
            return [min(lows), max(highs)]
        except TypeError:
            # domains of different kinds, e.g. a date and a number
            return None

    def _inspect(self, plot: Any) -> alt.TopLevelMixin:
        """
        Return the chart of a plot, for reading only, built at most once.

        :param plot: a ``Plot`` or an Altair chart
        :return: Altair chart
        """
        key = id(plot)
        chart = self._inspected.get(key)
        if chart is None:
            chart = _panel_chart(plot)
            self._inspected[key] = chart
        return chart

    # ------------------------------------------------------------------ sizing

    def _extents(
        self,
        node: Group,
        context: _Extent,
        spacing: int,
        vertical: bool,
    ) -> List[Optional[int]]:
        """
        Resolve the extent of every child along the stacking axis.

        :param node: the group whose children are measured
        :param context: the sizing context of the group
        :param spacing: the inter-panel gap of the group
        :param vertical: whether the group stacks vertically
        :return: list of int or None
        """
        total = context.height if vertical else context.width
        default = context.panel_height if vertical else context.panel_width
        axis = "height" if vertical else "width"
        children = node.children
        count = len(children)

        extents: List[Optional[int]] = [None] * count
        weights: Dict[int, float] = {}
        fixed = 0
        for index, child in enumerate(children):
            size = child.size
            if isinstance(size, Px):
                extents[index] = int(size)
                fixed += int(size)
            elif size is None:
                weights[index] = 1.0
            else:
                weights[index] = float(size)

        if not weights:
            return extents

        if len(weights) == count and set(weights.values()) == {1.0}:
            # no track was sized, so every panel keeps the extent of the figure
            return extents

        if all(weight == int(weight) for weight in weights.values()):
            # integer weights count grid tracks, they are not a share of a total :
            # a panel spanning two tracks is twice as long, gap included
            unit = default
            if unit is None:
                unit = DEFAULT_TRACK_HEIGHT if vertical else DEFAULT_TRACK_WIDTH
            for index, weight in weights.items():
                tracks = int(weight)
                extents[index] = unit * tracks + spacing * (tracks - 1)
            return extents

        if total is not None:
            # the room left by the gaps and by the panels sized in pixels is
            # split between the weighted panels, proportionally to their weight
            available = total - spacing * max(count - 1, 0) - fixed
            share = sum(weights.values())
            for index, weight in weights.items():
                extents[index] = max(int(round(weight / share * available)), 1)
            return extents

        if not self.build:
            # validation only, the extent may still be set afterwards
            return [default if extent is None else extent for extent in extents]
        raise ValueError(
            f"fractional sizes need the total {axis} of the figure, "
            f"call .properties({axis}=...) or give pixel sizes"
        )

    def _cross_extent(
        self, node: Group, context: _Extent, vertical: bool
    ) -> Optional[int]:
        """
        Resolve the extent handed to the children across the stacking axis.

        A group spans as many tracks across its stacking axis as its widest child
        does, so the extent handed over is the extent of those tracks. Otherwise the
        extent of the figure is handed over, and columns of unequal heights would be
        misaligned, so when the figure has no explicit height the natural height of
        the group is measured from the panels and their pixel extents instead.

        :param node: the group whose children are measured
        :param context: the sizing context of the group
        :param vertical: whether the group stacks vertically
        :return: int or None
        """
        axis = "x" if vertical else "y"
        unit = context.panel_width if vertical else context.panel_height
        tracks = self._cross_tracks(node, axis)
        if unit is not None and tracks is not None and tracks > 1:
            spacing = _resolve_spacing(node.spacing, self.figure._spacing)
            return unit * tracks + spacing * (tracks - 1)
        total = context.width if vertical else context.height
        if total is not None:
            return total
        if unit is not None:
            return unit
        return None if vertical else self._natural(node, "y")

    def _cross_tracks(self, node: LayoutNode, axis: str) -> Optional[int]:
        """
        Count the grid tracks a subtree spans along an axis, if they are all unsized.

        :param node: the node to measure
        :param axis: ``"x"`` or ``"y"``
        :return: int, or None when a track of the subtree is sized
        """
        if not isinstance(node, Group):
            return 1
        if node._axis != axis:
            # the tracks of the axis are the ones of the children, and a child
            # spanning several of them has a sibling covering the same tracks
            counts = [self._cross_tracks(child, axis) for child in node.children]
            if any(count is None for count in counts):
                return None
            return max(counts)
        tracks = 0
        for child in node.children:
            size = child.size
            if size is None:
                size = 1
            elif isinstance(size, Px) or size != int(size):
                return None
            tracks += int(size)
        return tracks

    def _natural(self, node: LayoutNode, axis: str) -> Optional[int]:
        """
        Measure the extent a subtree takes along an axis, if it is known.

        :param node: the node to measure
        :param axis: ``"x"`` or ``"y"``
        :return: int or None
        """
        if isinstance(node, Spacer):
            return None
        if isinstance(node, Panel):
            width, height = _explicit_dims(self._inspect(node.plot))
            return width if axis == "x" else height

        spacing = _resolve_spacing(node.spacing, self.figure._spacing)
        along = node._axis == axis
        found = [
            (
                int(child.size)
                if along and isinstance(child.size, Px)
                else self._natural(child, axis)
            )
            for child in node.children
        ]
        if along:
            if any(extent is None for extent in found):
                return None
            return sum(found) + spacing * (len(found) - 1)
        known = [extent for extent in found if extent is not None]
        return max(known) if known else None

    def _child_context(
        self,
        child: LayoutNode,
        context: _Extent,
        extent: Optional[int],
        cross: Optional[int],
        spacing: int,
        vertical: bool,
    ) -> _Extent:
        """
        Build the sizing context of a child from the one of its parent.

        The extent given to a child is a total, hence a child stacking along the
        same axis splits it between its own children, whereas any other child hands
        it over as is. Across the stacking axis, a panel fills the extent of the
        tracks it spans, while a group hands the track extent over untouched, which
        is what keeps its panels aligned with the ones of its siblings.

        :param child: the child node
        :param context: the sizing context of the parent
        :param extent: the extent given to the child along the stacking axis
        :param cross: the extent given to the child across the stacking axis
        :param spacing: the inter-panel gap of the parent
        :param vertical: whether the parent stacks vertically
        :return: _Extent
        """
        same_axis = isinstance(child, Rows if vertical else Cols)
        panel_extent = extent
        if extent is None:
            # an unsized track keeps the panel extent of the figure
            panel_extent = context.panel_height if vertical else context.panel_width
        elif same_axis:
            panel_extent = _split(extent, len(child.children), spacing)

        if isinstance(child, Group):
            panel_cross = context.panel_width if vertical else context.panel_height
        else:
            # a panel fills the tracks it spans across the stacking axis
            panel_cross = cross

        if vertical:
            return _Extent(
                width=cross,
                height=extent,
                panel_width=panel_cross,
                panel_height=panel_extent,
            )
        return _Extent(
            width=extent,
            height=cross,
            panel_width=panel_extent,
            panel_height=panel_cross,
        )

    def _check_equal_widths(self, node: Group, context: _Extent) -> None:
        """
        Check that panels stacked under a shared x axis can be aligned.

        :param node: the group to check
        :param context: the sizing context of the group
        :return: None
        """
        if context.panel_width is not None:
            # handled panel by panel, against the resolved target width
            return
        widths = set()
        for panel in node.panels():
            width, _ = _explicit_dims(self._inspect(panel.plot))
            widths.add(width)
        if len(widths) > 1:
            raise ValueError(
                f"panels stacked under a shared x axis have different widths "
                f"{sorted(w for w in widths if w is not None)}, set a single "
                f"width with Figure.properties(width=...)"
            )


# --------------------------------------------------------------------- factories


def layout(
    spec: str,
    panels: Optional[Dict[str, Panelish]] = None,
    /,
    *,
    widths: Optional[Sequence[Number]] = None,
    heights: Optional[Sequence[Number]] = None,
    share_x: Optional[Sharing] = None,
    share_y: Optional[Sharing] = None,
    labels: Optional[str] = None,
    spacing: Optional[int] = None,
    title: Optional[str] = None,
    **named: Panelish,
) -> Figure:
    """
    Build a figure from a picture of it, one character per cell.

        on.layout(
            '''
            A A B
            C C B
            ''',
            A=solar, B=nuclear, C=total,
        )

    Every character names a panel, a repeated character spans the cells it
    covers, and ``.`` leaves a gap. Whitespace and indentation are meaningless.
    The string holds the geometry only, there is no size and no operator in it,
    extents are given by ``widths`` and ``heights``.

    :param spec: the layout string
    :param panels: the panels, as a mapping of character to panel, an
        alternative to the keyword form
    :param widths: one extent per grid column, all ints (px) or all floats
        (relative weights)
    :param heights: one extent per grid row, same units as ``widths``
    :param share_x: whether the x scale domain is shared, defaults to ``True``.
        Either a boolean, or named groups given as a string of characters
        (``"AC"``) or a list of them (``["AC", "BD"]``), the panels outside of
        every group keeping an independent scale.
    :param share_y: whether the y scale domain is shared, defaults to ``False``,
        same forms as ``share_x``
    :param labels: ``"bottom"`` (default) to draw the inner x axis labels only on
        the bottom row of a group sharing its x scale, ``"all"`` to draw them on
        every panel
    :param spacing: inter-panel gap in px, defaults to 4
    :param title: figure title
    :return: Figure
    """
    if panels is not None and named:
        raise TypeError(
            "give the panels either as a mapping or as keyword arguments, " "not both"
        )
    mapping = dict(panels) if panels is not None else dict(named)
    if not mapping:
        raise ValueError(
            'a layout needs its panels, e.g. on.layout("A B", A=first, B=second)'
        )
    return _figure(
        _Source(
            kind="layout",
            spec=spec,
            panels=mapping,
            widths=widths,
            heights=heights,
            share_x=share_x,
            share_y=share_y,
            labels=labels,
            spacing=spacing,
            title=title,
        )
    )


def rows(
    *panels: Panelish,
    share_x: Optional[Sharing] = None,
    share_y: Optional[Sharing] = None,
    heights: Optional[Sequence[Number]] = None,
    labels: Optional[str] = None,
    spacing: Optional[int] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Stack panels vertically, i.e. the given panels are the rows of the figure.

    :param panels: the panels, either ``Plot``, ``Figure`` or Altair charts
    :param share_x: whether the x scale domain is shared, defaults to ``True``,
        inner x axis labels are then hidden and only drawn on the bottom row.
        Left unset, a ``share_x`` given explicitly by an enclosing group is
        inherited instead of the default. Named groups are accepted too, see
        :func:`layout`.
    :param share_y: whether the y scale domain is shared, defaults to ``False``
        since stacked panels usually carry different units. Left unset, a
        ``share_y`` given explicitly by an enclosing group is inherited instead
        of the default.
    :param heights: one extent per row, either a list of ints (px) or a list of
        floats taken as relative weights
    :param labels: ``"bottom"`` (default) or ``"all"``, see :func:`layout`
    :param spacing: inter-panel gap in px, defaults to 4
    :param title: title of the group
    :return: Figure
    """
    return _build(
        "rows", panels, share_x, share_y, None, heights, labels, spacing, title
    )


def cols(
    *panels: Panelish,
    share_x: Optional[Sharing] = None,
    share_y: Optional[Sharing] = None,
    widths: Optional[Sequence[Number]] = None,
    labels: Optional[str] = None,
    spacing: Optional[int] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Place panels side by side, i.e. the given panels are the columns.

    :param panels: the panels, either ``Plot``, ``Figure`` or Altair charts
    :param share_x: whether the x scale domain is shared, defaults to ``False``
        since columns usually show different periods. Left unset, a ``share_x``
        given explicitly by an enclosing group is inherited instead of the
        default. Named groups are accepted too, see :func:`layout`.
    :param share_y: whether the y scale domain is shared, defaults to
        ``False``. Left unset, a ``share_y`` given explicitly by an enclosing
        group is inherited instead of the default.
    :param widths: one extent per column, either a list of ints (px) or a list of
        floats taken as relative weights
    :param labels: ``"bottom"`` (default) or ``"all"``, see :func:`layout`
    :param spacing: inter-panel gap in px, defaults to 4
    :param title: title of the group
    :return: Figure
    """
    return _build(
        "cols", panels, share_x, share_y, widths, None, labels, spacing, title
    )


def grid(
    panels: Sequence[Panelish],
    columns: int,
    *,
    share_x: Optional[Sharing] = None,
    share_y: Optional[Sharing] = None,
    widths: Optional[Sequence[Number]] = None,
    heights: Optional[Sequence[Number]] = None,
    labels: Optional[str] = None,
    spacing: Optional[int] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Wrap panels into a regular grid of the given number of columns.

    The last row is padded with gaps when the panels do not fill it, so the
    panels of the other rows keep their width.

    :param panels: the panels, in reading order
    :param columns: the number of columns of the grid
    :param share_x: whether the x scale domain is shared, defaults to ``True``
    :param share_y: whether the y scale domain is shared, defaults to ``True``
        since a grid usually shows comparable panels
    :param widths: one extent per grid column
    :param heights: one extent per grid row
    :param labels: ``"bottom"`` (default) or ``"all"``, see :func:`layout`
    :param spacing: inter-panel gap in px, defaults to 4
    :param title: title of the group
    :return: Figure
    """
    if isinstance(panels, (str, bytes)) or not isinstance(panels, AbcSequence):
        raise TypeError(
            f"grid expects a sequence of panels, got {type(panels).__name__}"
        )
    panels = tuple(panels)
    if not panels:
        raise ValueError("grid needs at least one panel, none was given")
    if not isinstance(columns, int) or isinstance(columns, bool) or columns < 1:
        raise ValueError(f"columns must be a strictly positive int, got {columns!r}")

    names = _names(len(panels))
    cells = [*names, *[_grid.GAP] * (-len(panels) % columns)]
    spec = "\n".join(
        " ".join(cells[start : start + columns])
        for start in range(0, len(cells), columns)
    )
    return _figure(
        _Source(
            kind="grid",
            spec=spec,
            panels=dict(zip(names, panels)),
            widths=widths,
            heights=heights,
            share_x=share_x,
            share_y=share_y,
            labels=labels,
            spacing=spacing,
            title=title,
        )
    )


def _build(
    kind: str,
    panels: Tuple[Panelish, ...],
    share_x: Optional[Sharing],
    share_y: Optional[Sharing],
    widths: Optional[Sequence[Number]],
    heights: Optional[Sequence[Number]],
    labels: Optional[str],
    spacing: Optional[int],
    title: Optional[str],
) -> Figure:
    """
    Build a figure from a flat group of panels.

    :param kind: ``"rows"`` or ``"cols"``
    :param panels: the panels
    :param share_x: whether the x scale domain is shared
    :param share_y: whether the y scale domain is shared
    :param widths: one extent per grid column
    :param heights: one extent per grid row
    :param labels: the labels policy of the group
    :param spacing: inter-panel gap in px
    :param title: title of the group
    :return: Figure
    """
    if not panels:
        raise ValueError(f"{kind} needs at least one panel, none was given")
    names = _names(len(panels))
    separator = "\n" if kind == "rows" else " "
    return _figure(
        _Source(
            kind=kind,
            spec=separator.join(names),
            panels=dict(zip(names, panels)),
            widths=widths,
            heights=heights,
            share_x=share_x,
            share_y=share_y,
            labels=labels,
            spacing=spacing,
            title=title,
        )
    )


def _names(count: int) -> Tuple[str, ...]:
    """
    Return the grid characters naming a given number of panels.

    :param count: the number of panels
    :return: tuple of str
    """
    if count > len(_grid.ALPHABET):
        raise ValueError(
            f"a figure holds at most {len(_grid.ALPHABET)} panels, got {count}"
        )
    return tuple(_grid.ALPHABET[:count])


class _Source:
    """
    How a figure was built, kept so that its layout can be built again.

    Track extents and scale sharing groups are resolved against the grid, hence
    a figure keeps the grid it came from instead of only its layout tree.

    :param kind: ``"layout"``, ``"rows"``, ``"cols"`` or ``"grid"``
    :param spec: the layout string of the figure
    :param panels: the panels, as a mapping of grid character to panel
    :param widths: one extent per grid column
    :param heights: one extent per grid row
    :param share_x: the x sharing flag or groups
    :param share_y: the y sharing flag or groups
    :param labels: the labels policy
    :param spacing: inter-panel gap in px
    :param title: title of the group
    """

    __slots__ = (
        "kind",
        "spec",
        "panels",
        "widths",
        "heights",
        "share_x",
        "share_y",
        "labels",
        "spacing",
        "title",
    )

    def __init__(
        self,
        kind: str,
        spec: str,
        panels: Dict[str, Panelish],
        widths: Optional[Sequence[Number]] = None,
        heights: Optional[Sequence[Number]] = None,
        share_x: Optional[Sharing] = None,
        share_y: Optional[Sharing] = None,
        labels: Optional[str] = None,
        spacing: Optional[int] = None,
        title: Optional[str] = None,
    ):
        self.kind = kind
        self.spec = spec
        self.panels = panels
        self.widths = widths
        self.heights = heights
        self.share_x = share_x
        self.share_y = share_y
        self.labels = labels
        self.spacing = spacing
        self.title = title

    def replace(self, **changes: Any) -> "_Source":
        """
        Return a copy of the source with the given fields changed.

        :param changes: the fields to change, a ``None`` value keeping the
            current one
        :return: _Source
        """
        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(
            {name: value for name, value in changes.items() if value is not None}
        )
        return _Source(**values)


def _figure(source: _Source) -> Figure:
    """
    Build the figure of a source.

    :param source: how the figure is described
    :return: Figure
    """
    node, chars = _assemble(source)
    figure = Figure(node, source)
    figure._chars = chars
    return figure


def _assemble(
    source: _Source, spacing: Optional[int] = None
) -> Tuple[LayoutNode, Dict[str, Tuple[Panel, ...]]]:
    """
    Build the layout tree of a source.

    :param source: how the figure is described
    :param spacing: the gap a panel spanning several tracks also covers, only
        used when the source has no spacing of its own
    :return: tuple of the layout tree and of a mapping of grid character to the
        panels it names
    """
    rows_count, columns_count = _grid.shape(source.spec)
    heights = normalise_tracks(source.heights, rows_count, "y")
    widths = normalise_tracks(source.widths, columns_count, "x")
    for candidate in (source.spacing, spacing, DEFAULT_SPACING):
        if candidate is not None:
            spacing = candidate
            break

    skeleton = _grid.parse(source.spec, widths, heights, spacing)
    expected = set(_grid.characters(source.spec))
    given = set(source.panels)
    missing = sorted(expected - given)
    unused = sorted(given - expected)
    if missing:
        raise ValueError(
            f"the layout uses the character(s) {', '.join(missing)} but no panel "
            f"was given for them, the panels given are "
            f"{', '.join(sorted(given)) or 'none'}"
        )
    if unused:
        raise ValueError(
            f"the panel(s) {', '.join(unused)} do not appear in the layout, "
            f"which uses {', '.join(sorted(expected))}"
        )

    nodes = {char: _to_node(panel) for char, panel in source.panels.items()}
    node, chars = _grid.bind(skeleton, nodes)

    share_x = _resolve_sharing(source.share_x, "share_x", source, chars)
    share_y = _resolve_sharing(source.share_y, "share_y", source, chars)
    labels = _check_labels(source.labels)

    # the sharing a front end defaults to is recorded on the group, so that it
    # also reaches the groups nested under it, whereas the axes it leaves alone
    # stay open to the sharing of an enclosing group
    kind = _KIND_SHARING[source.kind]
    group = type(node) if isinstance(node, Group) else Rows
    if share_x is None and kind["x"]:
        share_x = True
    if share_y is None and kind["y"]:
        share_y = True

    children = node.children if isinstance(node, Group) else (node,)
    node = group(
        children,
        node.size,
        share_x=share_x,
        share_y=share_y,
        spacing=source.spacing,
        title=source.title,
        labels=labels,
    )
    return node, chars


def _check_labels(labels: Optional[str]) -> Optional[str]:
    """
    Validate a labels policy.

    :param labels: ``"bottom"``, ``"all"`` or None
    :return: str or None
    """
    if labels is None:
        return None
    if labels not in ("bottom", "all"):
        raise ValueError(f"labels must be 'bottom' or 'all', got {labels!r}")
    return labels


def _resolve_sharing(
    value: Optional[Sharing],
    name: str,
    source: _Source,
    chars: Dict[str, Tuple[Panel, ...]],
) -> Sharing:
    """
    Resolve a sharing argument to a boolean or to explicit groups of panels.

    A group is either a string of grid characters, e.g. ``"AC"``, or a sequence
    of panel objects. A sequence of strings, or of sequences of panels, describes
    several groups.

    :param value: the sharing argument
    :param name: the name of the argument, for the error messages
    :param source: how the figure is described
    :param chars: a mapping of grid character to the panels it names
    :return: bool, ShareGroups or None
    """
    if value is None or isinstance(value, bool):
        return value

    if isinstance(value, str):
        groups: List[Any] = [value]
    elif isinstance(value, AbcSequence):
        groups = list(value)
        if not groups:
            raise ValueError(f"{name} was given no group")
        if not all(isinstance(group, (str, AbcSequence)) for group in groups):
            # a flat sequence of panels is a single group
            groups = [groups]
    else:
        raise TypeError(
            f"{name} must be a boolean, a string of grid characters or a list "
            f"of groups, got {type(value).__name__}"
        )

    resolved = []
    for group in groups:
        members: List[Panel] = []
        for member in group:
            members.extend(_group_member(member, name, source, chars))
        if members:
            resolved.append(tuple(members))
    return ShareGroups(resolved)


def _group_member(
    member: Any,
    name: str,
    source: _Source,
    chars: Dict[str, Tuple[Panel, ...]],
) -> Tuple[Panel, ...]:
    """
    Resolve one member of a sharing group to the panels it names.

    :param member: a grid character or a panel object
    :param name: the name of the sharing argument, for the error messages
    :param source: how the figure is described
    :param chars: a mapping of grid character to the panels it names
    :return: tuple of Panel
    """
    if isinstance(member, str):
        if member not in chars:
            raise ValueError(
                f"{name} names the panel {member!r}, which is not in the layout, "
                f"the panels are {', '.join(sorted(chars))}"
            )
        return chars[member]
    for char, panel in source.panels.items():
        if panel is member:
            return chars[char]
    raise ValueError(
        f"{name} names a panel that is not part of the figure, give one of its "
        f"panels or the grid characters naming them, e.g. {name}='AB'"
    )


def _to_node(panel: Panelish) -> LayoutNode:
    """
    Coerce a panel to a layout node.

    Figure level properties of a nested figure are dropped, only its layout is
    kept, panel level properties are preserved.

    :param panel: a ``Plot``, a ``Figure`` or an Altair chart
    :return: LayoutNode
    """
    from .plot import Plot

    if isinstance(panel, Figure):
        return panel.layout
    if isinstance(panel, LayoutNode):
        return panel
    if isinstance(panel, (Plot, alt.TopLevelMixin)):
        return Panel(panel)
    raise TypeError(
        f"panels must be Plot or Figure instances, got {type(panel).__name__}"
    )


# ----------------------------------------------------------------------- helpers


def _walk(node: LayoutNode) -> List[LayoutNode]:
    """
    Return every node of a layout tree, parents first.

    :param node: the root of the tree
    :return: list of LayoutNode
    """
    found = [node]
    for child in getattr(node, "children", ()):
        found.extend(_walk(child))
    return found


def _channel_domain(chart: Any, channel_name: str) -> Optional[Tuple[Any, Any]]:
    """
    Read the data domain of a channel of a chart, ``None`` when unknown.

    The domain is read from the data frame behind the chart, before any data
    transformer runs, and temporal bounds are turned into ISO strings, which is
    the form Vega-Lite expects in a scale domain.

    :param chart: an Altair chart
    :param channel_name: ``"x"`` or ``"y"``
    :return: tuple of the lowest and the highest value, or None
    """
    import pandas as pd

    lows: List[Any] = []
    highs: List[Any] = []

    def visit(spec: Any, inherited: Any) -> None:
        data = _field(spec, "data")
        if data is alt.Undefined:
            data = inherited
        encoding = _field(spec, "encoding")
        channel = (
            _field(encoding, channel_name)
            if encoding is not alt.Undefined
            else alt.Undefined
        )
        field = _channel_field(channel)
        if field is not None and isinstance(data, pd.DataFrame) and field in data:
            values = data[field].dropna()
            if len(values):
                lows.append(values.min())
                highs.append(values.max())
        for layer in _sub_specs(spec):
            visit(layer, data)

    visit(chart, alt.Undefined)
    if not lows:
        return None

    def bound(value: Any) -> Any:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
        return value

    try:
        return bound(min(lows)), bound(max(highs))
    except TypeError:
        return None


def _channel_field(channel: Any) -> Optional[str]:
    """
    Read the data field a channel encodes, ``None`` when it has none.

    :param channel: an Altair channel
    :return: str or None
    """
    if channel is alt.Undefined or channel is None:
        return None
    for name in ("field", "shorthand"):
        value = _field(channel, name)
        if isinstance(value, str) and value:
            # a shorthand carries the type, and possibly an aggregate
            field = value.split(":")[0].strip()
            if field.endswith(")") and "(" in field:
                field = field[field.index("(") + 1 : -1].strip()
            return field or None
    return None


def _pin_domain(chart: Any, channel_name: str, domain: List[Any]) -> None:
    """
    Pin the scale domain of a channel of a chart, in place.

    Used for the panels of a named sharing group, whose members are not
    necessarily a subtree of the layout and therefore cannot be linked with the
    Vega-Lite ``resolve`` mechanism.

    :param chart: an Altair chart, already copied
    :param channel_name: ``"x"`` or ``"y"``
    :param domain: the domain to pin
    :return: None
    """
    encoding = _field(chart, "encoding")
    channel = (
        _field(encoding, channel_name)
        if encoding is not alt.Undefined
        else alt.Undefined
    )
    if channel is not alt.Undefined and channel is not None:
        scale = _field(channel, "scale")
        if scale is alt.Undefined or scale is None:
            channel["scale"] = alt.Scale(domain=domain)
        else:
            scale = scale.copy(deep=True)
            scale["domain"] = domain
            channel["scale"] = scale
    for layer in _sub_specs(chart):
        _pin_domain(layer, channel_name, domain)


def _resolve_flag(flag: Sharing, inherited: bool) -> Sharing:
    """
    Resolve a sharing flag, an explicit value winning over the inherited one.

    :param flag: the flag of the node, ``None`` meaning "inherit"
    :param inherited: the value coming from the enclosing group
    :return: bool or ShareGroups
    """
    if flag is None:
        return inherited
    if isinstance(flag, (bool, ShareGroups)):
        return flag
    return bool(flag)


def _resolve_spacing(spacing: Optional[int], figure_spacing: Optional[int]) -> int:
    """
    Resolve the gap of a group, a group value winning over the figure one.

    :param spacing: the spacing of the group
    :param figure_spacing: the spacing of the figure
    :return: int
    """
    if spacing is not None:
        return spacing
    if figure_spacing is not None:
        return figure_spacing
    return DEFAULT_SPACING


def _split(extent: int, count: int, spacing: int) -> int:
    """
    Split a total extent between the panels sharing it.

    :param extent: the total extent in px
    :param count: the number of panels
    :param spacing: the inter-panel gap in px
    :return: int
    """
    available = extent - spacing * max(count - 1, 0)
    return max(int(available // max(count, 1)), 1)


def _panel_chart(plot: Any) -> alt.TopLevelMixin:
    """
    Build the Altair chart of a panel, leaving the panel untouched.

    :param plot: a ``Plot`` or an Altair chart
    :return: Altair chart
    """
    from .plot import Plot

    if isinstance(plot, Plot):
        if not plot.layers:
            raise ValueError(
                "a panel needs at least one mark, add one with Plot.add(...)"
            )
        chart = plot.show().copy(deep=True)
    else:
        chart = plot.copy(deep=True)
    _hoist_title(chart)
    return chart


def _hoist_title(chart: Any) -> None:
    """
    Move the title of a lone layer up to the panel, in place.

    A title set through ``Plot.properties(title=...)`` lands on a layer, where it
    is drawn inside the panel and therefore sits at a slightly different place in
    every panel. Moving it to the panel keeps the titles of a group aligned.

    :param chart: an Altair chart, already copied
    :return: None
    """
    layers = _sub_specs(chart)
    if not layers or _field(chart, "title") is not alt.Undefined:
        return
    titled = [
        layer for layer in layers if _field(layer, "title") not in (alt.Undefined, None)
    ]
    if len(titled) != 1:
        # several titles are kept where they are, they name marks, not the panel
        return
    chart["title"] = _field(titled[0], "title")
    titled[0]["title"] = alt.Undefined


def _field(spec: Any, name: str) -> Any:
    """
    Read a schema field of an Altair object, ``alt.Undefined`` when unset.

    Plain attribute access cannot be used since Altair returns property setters
    for some channel attributes.

    :param spec: an Altair object
    :param name: the name of the field
    :return: the value of the field or ``alt.Undefined``
    """
    getter = getattr(spec, "_get", None)
    if getter is not None:
        return getter(name)
    return getattr(spec, name, alt.Undefined)


def _explicit_dims(
    chart: alt.TopLevelMixin,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Read the width and height explicitly set on a chart, if any.

    :param chart: an Altair chart
    :return: tuple of the width and the height, either int or None
    """

    def dimension(spec: Any, name: str) -> Optional[int]:
        value = _field(spec, name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        for layer in _sub_specs(spec):
            found = dimension(layer, name)
            if found is not None:
                return found
        return None

    return dimension(chart, "width"), dimension(chart, "height")


def _sub_specs(spec: Any) -> List[Any]:
    """
    Return the sub-specifications of a chart, if any.

    :param spec: an Altair chart
    :return: list of Altair charts
    """
    layers = _field(spec, "layer")
    if isinstance(layers, list):
        return layers
    return []


def _hide_axis(chart: Any, channel_name: str) -> None:
    """
    Hide the labels, ticks and title of an axis of a chart, in place.

    Used on the panels of a group sharing a scale, so that the axis is drawn only
    once, on the bottom row for x and on the leftmost column for y. The extent of
    the hidden axis is zeroed as well, so that it leaves no gap between panels.

    :param chart: an Altair chart, already copied
    :param channel_name: ``"x"`` or ``"y"``
    :return: None
    """
    encoding = _field(chart, "encoding")
    channel = (
        _field(encoding, channel_name)
        if encoding is not alt.Undefined
        else alt.Undefined
    )
    if channel is not alt.Undefined and channel is not None:
        axis = _field(channel, "axis")
        if axis is None:
            # the axis is already hidden altogether
            pass
        elif axis is alt.Undefined:
            channel["axis"] = alt.Axis(**_HIDDEN_AXIS)
        else:
            axis = axis.copy(deep=True)
            for name, value in _HIDDEN_AXIS.items():
                axis[name] = value
            channel["axis"] = axis
    for layer in _sub_specs(chart):
        _hide_axis(layer, channel_name)


def _concatenate(
    concatenate: Any,
    charts: List[alt.TopLevelMixin],
    spacing: int,
    bounds: str = DEFAULT_BOUNDS,
) -> alt.TopLevelMixin:
    """
    Concatenate charts, passing only the layout properties Vega-Lite accepts.

    ``align`` is part of the general ``concat`` specification but not of
    ``vconcat`` / ``hconcat``, where alignment comes from equal panel extents and
    from a reserved axis extent. It is therefore only forwarded when supported.

    :param concatenate: ``alt.vconcat`` or ``alt.hconcat``
    :param charts: the charts to concatenate
    :param spacing: the inter-panel gap in px
    :param bounds: ``"full"`` or ``"flush"``
    :return: Altair chart
    """
    properties = {
        "bounds": bounds,
        "align": "each",
        "center": False,
        "spacing": spacing,
    }
    chart = concatenate(*charts)
    accepted = inspect.signature(type(chart).__init__).parameters
    for name, value in properties.items():
        if name in accepted:
            chart[name] = value
    return chart


def _apply_raw_resolve(
    chart: alt.TopLevelMixin, resolve: Dict[str, Dict[str, str]]
) -> alt.TopLevelMixin:
    """
    Forward a raw Vega-Lite resolve dict to a chart.

    :param chart: an Altair chart
    :param resolve: e.g. ``{"scale": {"x": "shared"}}``
    :return: Altair chart
    """
    for kind, mapping in resolve.items():
        method = getattr(chart, f"resolve_{kind}", None)
        if method is None:
            raise ValueError(
                f"unknown resolve entry '{kind}', expected one of "
                f"'scale', 'axis', 'legend'"
            )
        chart = method(**mapping)
    return chart
