import unittest

import altair as alt
import pandas as pd

import ontime as on
from ontime.core.plotting._layout import Cols, Panel, Rows


class TestFigure(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2024-01-01", periods=24, freq="h")
        self.a = self.make_plot("a", title="a")
        self.b = self.make_plot("b", title="b")
        self.c = self.make_plot("c", title="c")

    # ------------------------------------------------------------------ helpers

    def make_series(self, name):
        df = pd.DataFrame({name: range(len(self.index))}, index=self.index)
        df.index.name = "time"
        return on.TimeSeries.from_dataframe(df)

    def make_plot(self, name, title=None, **properties):
        plot = on.Plot(self.make_series(name)).add(on.marks.line)
        if title is not None:
            properties["title"] = title
        if properties:
            plot = plot.properties(**properties)
        return plot

    @staticmethod
    def compile(figure):
        """Compile a figure to a Vega-Lite dict, without the vegafusion transformer."""
        chart = figure.to_altair()
        alt.data_transformers.enable("default")
        return chart.to_dict()

    # ------------------------------------------------------- construction / IR

    def test_rows__nested_rows__should_be_flattened(self):
        self.assertIs(
            on.rows(on.rows(self.a, self.b), self.c).layout,
            on.rows(self.a, self.b, self.c).layout,
        )

    def test_rows__single_panel__should_collapse_to_the_panel(self):
        self.assertIs(on.rows(self.a).layout, Panel(self.a))

    def test_rows__same_layout_twice__should_be_the_same_ir_object(self):
        self.assertIs(
            on.rows(self.a, on.cols(self.b, self.c)).layout,
            on.rows(self.a, on.cols(self.b, self.c)).layout,
        )

    def test_layout__nested_cols__should_keep_orientation_nodes(self):
        layout = on.cols(on.rows(self.a, self.b), self.c).layout
        self.assertIsInstance(layout, Cols)
        self.assertIsInstance(layout.children[0], Rows)
        self.assertIsInstance(layout.children[1], Panel)

    def test_layout__cols_of_cols__should_be_flattened(self):
        self.assertIs(
            on.cols(on.cols(self.a, self.b), self.c).layout,
            on.cols(self.a, self.b, self.c).layout,
        )

    def test_layout__explicit_sharing_on_child__should_not_be_flattened(self):
        layout = on.rows(on.rows(self.a, self.b, share_y=True), self.c).layout
        self.assertEqual(len(layout.children), 2)

    def test_layout__node__should_be_immutable(self):
        layout = on.rows(self.a, self.b).layout
        with self.assertRaises(AttributeError):
            layout.share_x = True

    def test_repr__rows_of_two_panels__should_render_symbolically(self):
        self.assertEqual(repr(on.rows(self.a, self.b)), "a / b")

    def test_repr__nested_layout__should_only_bracket_when_needed(self):
        self.assertEqual(repr(on.cols(on.rows(self.a, self.b), self.c)), "a / b | c")
        self.assertEqual(repr(on.rows(on.cols(self.a, self.b), self.c)), "(a | b) / c")

    def test_repr__untitled_panels__should_fall_back_to_positions(self):
        self.assertEqual(
            repr(on.rows(self.make_plot("x"), self.make_plot("y"))), "p0 / p1"
        )

    def test_repr__sizes__should_be_annotated(self):
        figure = on.rows(self.a, self.b, sizes=[240, 40])
        self.assertEqual(repr(figure), "a:240 / b:40")

    # ------------------------------------------------------------- compilation

    def test_show__rows__should_return_a_vertical_concatenation(self):
        chart = on.rows(self.a, self.b).properties(width=800, height=140).show()
        self.assertIsInstance(chart, alt.VConcatChart)
        self.assertEqual(len(chart.vconcat), 2)

    def test_show__cols__should_return_an_horizontal_concatenation(self):
        chart = on.cols(self.a, self.b).properties(width=300).show()
        self.assertIsInstance(chart, alt.HConcatChart)

    def test_to_altair__rows__should_share_x_and_keep_y_independent(self):
        spec = self.compile(on.rows(self.a, self.b).properties(width=800, height=140))
        self.assertEqual(spec["resolve"]["scale"]["x"], "shared")
        self.assertEqual(spec["resolve"]["scale"]["y"], "independent")

    def test_to_altair__cols__should_keep_x_independent_by_default(self):
        spec = self.compile(on.cols(self.a, self.b).properties(width=300, height=200))
        self.assertEqual(spec["resolve"]["scale"]["x"], "independent")
        self.assertEqual(spec["resolve"]["scale"]["y"], "independent")

    def test_to_altair__cols_with_shared_y__should_share_y(self):
        figure = on.cols(self.a, self.b, share_y=True, share_x=False)
        spec = self.compile(figure.properties(width=300, height=200))
        self.assertEqual(spec["resolve"]["scale"]["y"], "shared")

    def test_to_altair__default_layout__should_flush_and_space_panels(self):
        spec = self.compile(on.rows(self.a, self.b).properties(width=800, height=140))
        self.assertEqual(spec["bounds"], "flush")
        self.assertEqual(spec["spacing"], 4)

    def test_to_altair__figure_spacing__should_override_the_default_gap(self):
        figure = on.rows(self.a, self.b).properties(width=800, height=140, spacing=12)
        self.assertEqual(self.compile(figure)["spacing"], 12)

    def test_to_altair__group_spacing__should_win_over_figure_spacing(self):
        figure = on.rows(self.a, self.b, spacing=20).properties(width=800, spacing=12)
        self.assertEqual(self.compile(figure)["spacing"], 20)

    def test_to_altair__shared_x__should_hide_inner_x_axis_labels(self):
        figure = on.rows(self.a, self.b, self.c).properties(width=800, height=140)
        panels = self.compile(figure)["vconcat"]
        axes = [panel["layer"][0]["encoding"]["x"].get("axis") for panel in panels]
        self.assertEqual(axes[0], {"labels": False, "title": None})
        self.assertEqual(axes[1], {"labels": False, "title": None})
        self.assertIsNone(axes[2])

    def test_to_altair__independent_x__should_keep_every_x_axis(self):
        figure = on.rows(self.a, self.b, share_x=False).properties(
            width=800, height=140
        )
        panels = self.compile(figure)["vconcat"]
        for panel in panels:
            self.assertIsNone(panel["layer"][0]["encoding"]["x"].get("axis"))

    def test_to_altair__figure_size__should_be_applied_to_every_panel(self):
        figure = on.rows(self.a, self.b).properties(width=800, height=140)
        panels = self.compile(figure)["vconcat"]
        self.assertEqual(
            [(panel["width"], panel["height"]) for panel in panels],
            [(800, 140), (800, 140)],
        )

    def test_to_altair__pixel_sizes__should_set_the_panel_heights(self):
        figure = on.rows(
            self.make_plot("main"), self.make_plot("strip"), sizes=[240, 40]
        ).properties(width=800)
        panels = self.compile(figure)["vconcat"]
        self.assertEqual([panel["height"] for panel in panels], [240, 40])
        self.assertEqual([panel["width"] for panel in panels], [800, 800])

    def test_to_altair__fractional_sizes__should_split_the_figure_height(self):
        figure = on.rows(
            self.make_plot("forecast"), self.make_plot("residuals"), sizes=[0.72, 0.28]
        ).properties(width=700, height=340)
        panels = self.compile(figure)["vconcat"]
        self.assertEqual([panel["height"] for panel in panels], [245, 95])

    def test_to_altair__panel_properties__should_win_over_figure_properties(self):
        figure = on.rows(
            self.make_plot("main", height=77), self.make_plot("strip")
        ).properties(width=800, height=140)
        panels = self.compile(figure)["vconcat"]
        self.assertEqual([panel["height"] for panel in panels], [77, 140])

    def test_to_altair__nested_groups__should_size_panels_along_both_axes(self):
        figure = on.cols(
            on.rows(self.make_plot("main"), self.make_plot("strip"), sizes=[240, 50]),
            self.make_plot("profile"),
            sizes=[620, 180],
        ).properties(height=300)
        spec = self.compile(figure)
        inner = spec["hconcat"][0]["vconcat"]
        self.assertEqual(
            [(panel["width"], panel["height"]) for panel in inner],
            [(620, 240), (620, 50)],
        )
        self.assertEqual(
            (spec["hconcat"][1]["width"], spec["hconcat"][1]["height"]), (180, 300)
        )

    def test_to_altair__nested_rows__should_use_its_own_sharing_default(self):
        figure = on.cols(
            on.rows(self.make_plot("main"), self.make_plot("strip")),
            self.make_plot("profile"),
        ).properties(width=300, height=200)
        spec = self.compile(figure)
        self.assertEqual(spec["resolve"]["scale"]["x"], "independent")
        self.assertEqual(spec["hconcat"][0]["resolve"]["scale"]["x"], "shared")

    def test_to_altair__explicit_sharing__should_propagate_to_nested_groups(self):
        figure = on.cols(
            on.rows(self.make_plot("main"), self.make_plot("strip")),
            self.make_plot("profile"),
            share_y=True,
        ).properties(width=300, height=200)
        spec = self.compile(figure)
        self.assertEqual(spec["hconcat"][0]["resolve"]["scale"]["y"], "shared")

    def test_to_altair__explicit_sharing_on_nested_group__should_win(self):
        figure = on.cols(
            on.rows(self.make_plot("main"), self.make_plot("strip"), share_y=False),
            self.make_plot("profile"),
            share_y=True,
        ).properties(width=300, height=200)
        spec = self.compile(figure)
        self.assertEqual(spec["hconcat"][0]["resolve"]["scale"]["y"], "independent")

    def test_to_altair__rows_of_cols__should_concatenate_rows_of_columns(self):
        panels = [self.make_plot(name, name) for name in ("a", "b", "c", "d", "e")]
        figure = on.rows(
            on.cols(*panels[:3]),
            on.cols(*panels[3:]),
            share_y=True,
        ).properties(width=180, height=90)
        spec = self.compile(figure)
        self.assertEqual(len(spec["vconcat"]), 2)
        self.assertEqual([len(row["hconcat"]) for row in spec["vconcat"]], [3, 2])
        self.assertEqual(spec["resolve"]["scale"]["y"], "shared")
        self.assertEqual(spec["vconcat"][0]["resolve"]["scale"]["y"], "shared")

    def test_to_altair__titles__should_be_set_on_group_and_figure(self):
        figure = on.rows(self.a, self.b, title="inner").properties(
            width=800, height=140, title="outer"
        )
        spec = self.compile(figure)
        self.assertEqual(spec["title"], "outer")

    def test_to_altair__resolve_escape_hatch__should_override_the_flags(self):
        figure = on.rows(self.a, self.b).properties(
            width=800, height=140, resolve={"scale": {"x": "independent"}}
        )
        spec = self.compile(figure)
        self.assertEqual(spec["resolve"]["scale"]["x"], "independent")

    def test_show__panels_given_as_altair_charts__should_be_accepted(self):
        chart = on.Plot(self.make_series("a")).add(on.marks.line).show()
        figure = on.rows(chart, chart).properties(width=400, height=100)
        self.assertIsInstance(figure.show(), alt.VConcatChart)

    def test_show__figure_of_figures__should_nest(self):
        inner = on.rows(self.a, self.b)
        figure = on.cols(inner, self.c).properties(width=300, height=100)
        chart = figure.show()
        self.assertIsInstance(chart, alt.HConcatChart)
        self.assertIsInstance(chart.hconcat[0], alt.VConcatChart)

    def test_repr_mimebundle__figure__should_render_without_show(self):
        figure = on.rows(self.a, self.b).properties(width=400, height=100)
        bundle = figure._repr_mimebundle_()
        bundle = bundle[0] if isinstance(bundle, tuple) else bundle
        self.assertIn("text/html", bundle)

    # ------------------------------------------------------------------ errors

    def test_rows__no_panel__should_raise(self):
        with self.assertRaises(ValueError):
            on.rows()

    def test_rows__unsupported_panel__should_raise(self):
        with self.assertRaises(TypeError):
            on.rows(self.a, "not a plot")

    def test_rows__sizes_of_wrong_length__should_raise(self):
        with self.assertRaises(ValueError):
            on.rows(self.a, self.b, sizes=[100, 200, 300])

    def test_rows__mixed_pixel_and_fractional_sizes__should_raise(self):
        with self.assertRaises(ValueError):
            on.rows(self.a, self.b, sizes=[240, 0.5])

    def test_rows__fractions_not_summing_to_one__should_raise(self):
        with self.assertRaises(ValueError):
            on.rows(self.a, self.b, sizes=[0.5, 0.3])

    def test_rows__negative_sizes__should_raise(self):
        with self.assertRaises(ValueError):
            on.rows(self.a, self.b, sizes=[240, -40])

    def test_to_altair__fractional_sizes_without_extent__should_raise(self):
        figure = on.rows(self.a, self.b, sizes=[0.5, 0.5]).properties(width=400)
        with self.assertRaises(ValueError):
            figure.to_altair()

    def test_properties__fractional_sizes_set_in_two_calls__should_pass(self):
        figure = on.rows(self.a, self.b, sizes=[0.6, 0.4])
        figure = figure.properties(width=400).properties(height=200)
        panels = self.compile(figure)["vconcat"]
        self.assertEqual([panel["height"] for panel in panels], [120, 80])

    def test_properties__panel_width_conflicting_with_figure__should_raise(self):
        figure = on.rows(self.make_plot("main", width=300), self.make_plot("strip"))
        with self.assertRaises(ValueError):
            figure.properties(width=500)

    def test_properties__stacked_panels_of_different_widths__should_raise(self):
        figure = on.rows(
            self.make_plot("main", width=300), self.make_plot("strip", width=500)
        )
        with self.assertRaises(ValueError):
            figure.properties(height=100)

    def test_properties__side_by_side_panels_of_different_widths__should_pass(self):
        figure = on.cols(
            self.make_plot("main", width=300), self.make_plot("profile", width=120)
        ).properties(height=100)
        self.assertIsInstance(figure.show(), alt.HConcatChart)

    def test_show__panel_without_mark__should_raise(self):
        figure = on.rows(on.Plot(self.make_series("a")), self.a)
        with self.assertRaises(ValueError):
            figure.show()

    def test_save__unsupported_extension__should_raise(self):
        figure = on.rows(self.a, self.b).properties(width=400, height=100)
        with self.assertRaises(ValueError):
            figure.save("figure.pdf")


if __name__ == "__main__":
    unittest.main()
