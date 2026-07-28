"""
Figure, the subplot container of onTime.

A ``Plot`` is a single panel, made of layered marks. A :class:`Figure` places
several panels next to each other. Figures are built with the factories
:func:`rows` and :func:`cols`, and they nest freely.

    import ontime as on

    on.rows(
        on.Plot(solar).add(on.marks.line),
        on.Plot(nuclear).add(on.marks.line),
    ).properties(width=800, height=140).show()

Naming : ``rows(a, b)`` reads as "a and b are rows", i.e. they are stacked
vertically. The factories describe their arguments, not the container, which
avoids the usual ``vstack`` / ``hstack`` ambiguity.

Scale sharing propagates : a ``share_x`` or ``share_y`` given **explicitly** to a
group is inherited by its nested groups, unless the nested call sets the flag
itself. A flag left unset nowhere in the chain falls back to the default of the
group kind, i.e. ``share_x=True`` and ``share_y=False`` for rows,
``share_x=False`` and ``share_y=False`` for cols. This is the most likely source
of surprise when nesting figures.

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

from ._layout import (
    DEFAULT_SPACING,
    Cols,
    Group,
    LayoutNode,
    Panel,
    Rows,
    normalise_sizes,
)

Panelish = Any  # Plot, Figure or Altair chart
Number = Union[int, float]

# Scale sharing defaults, applied when a flag can neither be read from the node
# nor inherited from an enclosing group.
_DEFAULT_SHARING = {
    Rows: {"x": True, "y": False},
    Cols: {"x": False, "y": False},
}


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
    """

    def __init__(self, layout: LayoutNode):
        if not isinstance(layout, LayoutNode):
            raise TypeError(
                f"Figure expects a layout node, got {type(layout).__name__}"
            )
        self._layout = layout
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._spacing: Optional[int] = None
        self._title: Optional[str] = None
        self._resolve: Optional[Dict[str, Dict[str, str]]] = None

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

    def properties(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        spacing: Optional[int] = None,
        title: Optional[str] = None,
        resolve: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> "Figure":
        """
        Set figure level properties.

        Figure level properties only fill values that panels left unset, panel
        level ``.properties()`` always wins. ``width`` and ``height`` are the
        default extents of a **single panel**, whereas fractional ``sizes``
        (see :func:`rows`) are fractions of these extents taken as the total
        extent of the group.

        :param width: default panel width in px
        :param height: default panel height in px
        :param spacing: inter-panel gap in px, defaults to 4
        :param title: figure title
        :param resolve: raw Vega-Lite resolve dict, e.g.
            ``{"scale": {"y": "independent"}, "legend": {"color": "shared"}}``,
            applied last and therefore overriding the ``share_x`` /
            ``share_y`` flags
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
        # panels of a shared x axis must be equally wide to stay aligned
        _Compiler(self, build=False).run()
        return self

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
        return repr(self._layout)

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

    def run(self) -> Optional[alt.TopLevelMixin]:
        """
        Compile the whole figure.

        :return: Altair chart or None when only validating
        """
        figure = self.figure
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
            align_width=False,
        )
        if not self.build:
            return None

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
        share_x: Optional[bool],
        share_y: Optional[bool],
        hide_x: bool,
        align_width: bool,
    ) -> Optional[alt.TopLevelMixin]:
        """
        Compile a single node of the layout tree.

        :param node: the node to compile
        :param context: the sizing context of the node
        :param share_x: x sharing explicitly set by an enclosing group, or None
        :param share_y: y sharing explicitly set by an enclosing group, or None
        :param hide_x: whether the x axis labels must be hidden
        :param align_width: whether the node is stacked under a shared x axis
        :return: Altair chart or None when only validating
        """
        if isinstance(node, Panel):
            return self._panel(node, context, hide_x, align_width)
        return self._group(node, context, share_x, share_y, hide_x, align_width)

    def _panel(
        self,
        node: Panel,
        context: _Extent,
        hide_x: bool,
        align_width: bool,
    ) -> Optional[alt.TopLevelMixin]:
        """
        Compile a leaf of the layout tree.

        :param node: the panel to compile
        :param context: the sizing context of the panel
        :param hide_x: whether the x axis labels must be hidden
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
        if hide_x:
            _hide_x_axis(chart)
        return chart

    def _group(
        self,
        node: Group,
        context: _Extent,
        share_x: Optional[bool],
        share_y: Optional[bool],
        hide_x: bool,
        align_width: bool,
    ) -> Optional[alt.TopLevelMixin]:
        """
        Compile a group of the layout tree.

        :param node: the group to compile
        :param context: the sizing context of the group
        :param share_x: x sharing explicitly set by an enclosing group, or None
        :param share_y: y sharing explicitly set by an enclosing group, or None
        :param hide_x: whether the x axis labels must be hidden
        :param align_width: whether the group is stacked under a shared x axis
        :return: Altair chart or None when only validating
        """
        # an explicit flag wins, then any flag explicitly set upstream, then the
        # default of the group kind
        default = _DEFAULT_SHARING[type(node)]
        explicit_x = node.share_x if node.share_x is not None else share_x
        explicit_y = node.share_y if node.share_y is not None else share_y
        group_share_x = _resolve_flag(explicit_x, default["x"])
        group_share_y = _resolve_flag(explicit_y, default["y"])
        spacing = _resolve_spacing(node.spacing, self.figure._spacing)

        vertical = isinstance(node, Rows)
        extents = self._extents(node, context, spacing, vertical)
        children_align = align_width or (vertical and group_share_x)

        if vertical and group_share_x:
            self._check_equal_widths(node, context)

        charts: List[alt.TopLevelMixin] = []
        last = len(node.children) - 1
        for index, child in enumerate(node.children):
            child_hide_x = hide_x or (vertical and group_share_x and index != last)
            chart = self._node(
                child,
                self._child_context(child, context, extents[index], spacing, vertical),
                share_x=explicit_x,
                share_y=explicit_y,
                hide_x=child_hide_x,
                align_width=children_align,
            )
            charts.append(chart)

        if not self.build:
            return None

        concatenate = alt.vconcat if vertical else alt.hconcat
        chart = _concatenate(concatenate, charts, spacing)
        chart = chart.resolve_scale(
            x="shared" if group_share_x else "independent",
            y="shared" if group_share_y else "independent",
        )
        if node.title is not None:
            chart = chart.properties(title=node.title)
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

        extents: List[Optional[int]] = []
        for child in node.children:
            size = child.size
            if size is None:
                extents.append(default)
            elif isinstance(size, float):
                if total is None and not self.build:
                    # validation only, the extent may still be set afterwards
                    extents.append(default)
                elif total is None:
                    raise ValueError(
                        f"fractional sizes need the total {axis} of the figure, "
                        f"call .properties({axis}=...) or give pixel sizes"
                    )
                else:
                    extents.append(int(round(size * total)))
            else:
                extents.append(size)
        return extents

    @staticmethod
    def _child_context(
        child: LayoutNode,
        context: _Extent,
        extent: Optional[int],
        spacing: int,
        vertical: bool,
    ) -> _Extent:
        """
        Build the sizing context of a child from the one of its parent.

        The extent given to a child is a total, hence a child stacking along the
        same axis splits it between its own children, whereas any other child
        hands it over as is.

        :param child: the child node
        :param context: the sizing context of the parent
        :param extent: the extent given to the child along the stacking axis
        :param spacing: the inter-panel gap of the parent
        :param vertical: whether the parent stacks vertically
        :return: _Extent
        """
        same_axis = isinstance(child, Rows if vertical else Cols)
        panel_extent = extent
        if same_axis and extent is not None:
            panel_extent = _split(extent, len(child.children), spacing)

        if vertical:
            return _Extent(
                width=context.width,
                height=extent,
                panel_width=context.panel_width,
                panel_height=panel_extent,
            )
        return _Extent(
            width=extent,
            height=context.height,
            panel_width=panel_extent,
            panel_height=context.panel_height,
        )

    @staticmethod
    def _check_equal_widths(node: Group, context: _Extent) -> None:
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
            width, _ = _explicit_dims(_panel_chart(panel.plot))
            widths.add(width)
        if len(widths) > 1:
            raise ValueError(
                f"panels stacked under a shared x axis have different widths "
                f"{sorted(w for w in widths if w is not None)}, set a single "
                f"width with Figure.properties(width=...)"
            )


# --------------------------------------------------------------------- factories


def rows(
    *panels: Panelish,
    share_x: Optional[bool] = None,
    share_y: Optional[bool] = None,
    sizes: Optional[Sequence[Number]] = None,
    spacing: Optional[int] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Stack panels vertically, i.e. the given panels are the rows of the figure.

    :param panels: the panels, either ``Plot``, ``Figure`` or Altair charts
    :param share_x: whether the x scale domain is shared, defaults to ``True``,
        inner x axis labels are then hidden and only drawn on the bottom row.
        Left unset, an ``share_x`` given explicitly by an enclosing group is
        inherited instead of the default.
    :param share_y: whether the y scale domain is shared, defaults to ``False``
        since stacked panels usually carry different units. Left unset, a
        ``share_y`` given explicitly by an enclosing group is inherited instead
        of the default.
    :param sizes: per-panel heights, either a list of ints (px) or a list of
        floats summing to 1.0 (fractions of the figure height)
    :param spacing: inter-panel gap in px, defaults to 4
    :param title: title of the group
    :return: Figure
    """
    return _build(Rows, panels, share_x, share_y, sizes, spacing, title)


def cols(
    *panels: Panelish,
    share_x: Optional[bool] = None,
    share_y: Optional[bool] = None,
    sizes: Optional[Sequence[Number]] = None,
    spacing: Optional[int] = None,
    title: Optional[str] = None,
) -> Figure:
    """
    Place panels side by side, i.e. the given panels are the columns.

    :param panels: the panels, either ``Plot``, ``Figure`` or Altair charts
    :param share_x: whether the x scale domain is shared, defaults to ``False``
        since columns usually show different periods. Left unset, a ``share_x``
        given explicitly by an enclosing group is inherited instead of the
        default.
    :param share_y: whether the y scale domain is shared, defaults to
        ``False``. Left unset, a ``share_y`` given explicitly by an enclosing
        group is inherited instead of the default.
    :param sizes: per-panel widths, either a list of ints (px) or a list of
        floats summing to 1.0 (fractions of the figure width)
    :param spacing: inter-panel gap in px, defaults to 4
    :param title: title of the group
    :return: Figure
    """
    return _build(Cols, panels, share_x, share_y, sizes, spacing, title)


def _build(
    kind: type,
    panels: Tuple[Panelish, ...],
    share_x: Optional[bool],
    share_y: Optional[bool],
    sizes: Optional[Sequence[Number]],
    spacing: Optional[int],
    title: Optional[str],
) -> Figure:
    """
    Build a figure from a group of panels.

    :param kind: ``Rows`` or ``Cols``
    :param panels: the panels
    :param share_x: whether the x scale domain is shared
    :param share_y: whether the y scale domain is shared
    :param sizes: per-panel extents along the stacking axis
    :param spacing: inter-panel gap in px
    :param title: title of the group
    :return: Figure
    """
    if not panels:
        raise ValueError(
            f"{kind.__name__.lower()} needs at least one panel, none was given"
        )
    nodes = [_to_node(panel) for panel in panels]
    checked = normalise_sizes(sizes, len(nodes), kind._axis)
    if checked is not None:
        nodes = [node.with_size(size) for node, size in zip(nodes, checked)]
    layout = kind(
        nodes,
        share_x=share_x,
        share_y=share_y,
        spacing=spacing,
        title=title,
    )
    return Figure(layout)


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


def _resolve_flag(flag: Optional[bool], inherited: bool) -> bool:
    """
    Resolve a sharing flag, an explicit value winning over the inherited one.

    :param flag: the flag of the node, ``None`` meaning "inherit"
    :param inherited: the value coming from the enclosing group
    :return: bool
    """
    return inherited if flag is None else bool(flag)


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
        return plot.show().copy(deep=True)
    return plot.copy(deep=True)


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


def _hide_x_axis(chart: Any) -> None:
    """
    Hide the x axis labels and title of a chart, in place.

    Used on the inner panels of a group sharing its x scale, so that the axis is
    only drawn once, on the bottom row.

    :param chart: an Altair chart, already copied
    :return: None
    """
    encoding = _field(chart, "encoding")
    channel = _field(encoding, "x") if encoding is not alt.Undefined else alt.Undefined
    if channel is not alt.Undefined and channel is not None:
        axis = _field(channel, "axis")
        if axis is None:
            # the axis is already hidden altogether
            pass
        elif axis is alt.Undefined:
            channel["axis"] = alt.Axis(labels=False, title=None)
        else:
            axis = axis.copy(deep=True)
            axis["labels"] = False
            axis["title"] = None
            channel["axis"] = axis
    for layer in _sub_specs(chart):
        _hide_x_axis(layer)


def _concatenate(
    concatenate: Any, charts: List[alt.TopLevelMixin], spacing: int
) -> alt.TopLevelMixin:
    """
    Concatenate charts, passing only the layout properties Vega-Lite accepts.

    ``align`` is part of the general ``concat`` specification but not of
    ``vconcat`` / ``hconcat``, where alignment comes from equal panel extents
    and from ``bounds="flush"``. It is therefore only forwarded when supported.

    :param concatenate: ``alt.vconcat`` or ``alt.hconcat``
    :param charts: the charts to concatenate
    :param spacing: the inter-panel gap in px
    :return: Altair chart
    """
    properties = {
        "bounds": "flush",
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
