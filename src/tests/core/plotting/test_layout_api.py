import unittest

import altair as alt
import pandas as pd

import ontime as on
from ontime.core.plotting._layout import Cols, Panel, Px, Rows, Spacer


class TestLayoutApi(unittest.TestCase):
    """Test the layout string front-end of the figure API."""

    def setUp(self):
        self.index = pd.date_range("2024-01-01", periods=24, freq="h")
        self.a = self.make_plot("a")
        self.b = self.make_plot("b")
        self.c = self.make_plot("c")
        self.d = self.make_plot("d")

    # ------------------------------------------------------------------ helpers

    def make_plot(self, name, offset=0, **properties):
        df = pd.DataFrame(
            {name: [value + offset for value in range(len(self.index))]},
            index=self.index,
        )
        df.index.name = "time"
        plot = on.Plot(on.TimeSeries.from_dataframe(df)).add(on.marks.line)
        return plot.properties(**properties) if properties else plot

    def compile(self, figure):
        """Compile a figure to a Vega-Lite dict, without the data transformer."""
        chart = figure.to_altair()
        alt.data_transformers.enable("default")
        try:
            return chart.to_dict()
        finally:
            alt.data_transformers.enable("vegafusion")

    @staticmethod
    def encoding(spec, channel):
        """Read a channel of a leaf spec, layered or not."""
        if "layer" in spec:
            spec = spec["layer"][0]
        return spec.get("encoding", {}).get(channel, {})

    # ------------------------------------------------------------------ binding

    def test_layout__keyword_panels__should_build_the_grid(self):
        figure = on.layout("A A B\nC C B", A=self.a, B=self.b, C=self.c)
        self.assertEqual(
            list(figure.layout.panels()),
            [Panel(self.a), Panel(self.c), Panel(self.b)],
        )

    def test_layout__mapping_of_panels__should_build_the_same_grid(self):
        keywords = on.layout("A B", A=self.a, B=self.b)
        mapping = on.layout("A B", {"A": self.a, "B": self.b})
        self.assertEqual(mapping.to_string(), keywords.to_string())
        self.assertIs(mapping.layout, keywords.layout)

    def test_layout__nested_figure__should_be_spliced_into_the_tree(self):
        figure = on.layout("A B", A=on.rows(self.a, self.b), B=self.c)
        self.assertEqual(figure.to_string(), "A B\nC B")

    def test_layout__unbound_character__should_raise(self):
        with self.assertRaises(ValueError) as raised:
            on.layout("A B", A=self.a)
        self.assertIn("B", str(raised.exception))

    def test_layout__unused_panel__should_raise(self):
        with self.assertRaises(ValueError) as raised:
            on.layout("A B", A=self.a, B=self.b, C=self.c)
        self.assertIn("C", str(raised.exception))

    def test_layout__panels_given_twice__should_raise(self):
        with self.assertRaises(TypeError):
            on.layout("A B", {"A": self.a}, B=self.b)

    def test_layout__not_a_plot__should_raise(self):
        with self.assertRaises(TypeError):
            on.layout("A B", A=self.a, B="b")

    # -------------------------------------------------------------- track sizes

    def test_layout__pixel_tracks__should_set_the_panel_extents(self):
        figure = on.layout(
            "A A B\nC C B",
            A=self.a,
            B=self.b,
            C=self.c,
            widths=[300, 300, 200],
            heights=[240, 40],
            spacing=4,
        )
        spec = self.compile(figure)
        panel_a = spec["hconcat"][0]["vconcat"][0]
        panel_b = spec["hconcat"][1]
        # a panel spanning two tracks also covers the gap between them
        self.assertEqual((panel_a["width"], panel_a["height"]), (604, 240))
        self.assertEqual((panel_b["width"], panel_b["height"]), (200, 284))

    def test_layout__weighted_tracks__should_split_the_figure_extent(self):
        figure = on.layout(
            "T T\nA B",
            T=self.a,
            A=self.b,
            B=self.c,
            heights=[0.7, 0.3],
            spacing=0,
        ).properties(width=600, height=300)
        heights = [row["height"] for row in self.compile(figure)["vconcat"][:1]]
        self.assertEqual(heights, [210])

    def test_layout__tracks_of_wrong_length__should_raise(self):
        with self.assertRaises(ValueError) as raised:
            on.layout("A B", A=self.a, B=self.b, widths=[100])
        self.assertIn("column", str(raised.exception))
        with self.assertRaises(ValueError) as raised:
            on.layout("A\nB", A=self.a, B=self.b, heights=[1.0, 1.0, 1.0])
        self.assertIn("row", str(raised.exception))

    def test_layout__mixed_track_units__should_raise(self):
        with self.assertRaises(ValueError):
            on.layout("A B", A=self.a, B=self.b, widths=[240, 0.5])

    def test_layout__non_positive_tracks__should_raise(self):
        for widths in ([240, 0], [240, -40]):
            with self.subTest(widths=widths):
                with self.assertRaises(ValueError):
                    on.layout("A B", A=self.a, B=self.b, widths=widths)

    def test_properties__tracks__should_rebuild_the_layout(self):
        figure = on.rows(self.a, self.b).properties(heights=[240, 40], width=800)
        heights = [row["height"] for row in self.compile(figure)["vconcat"]]
        self.assertEqual(heights, [240, 40])

    def test_properties__tracks_without_a_grid__should_raise(self):
        figure = on.Figure(Rows((Panel(self.a), Panel(self.b))))
        with self.assertRaises(ValueError):
            figure.properties(heights=[240, 40])

    # ----------------------------------------------------------------- rows API

    def test_rows__heights__should_size_the_rows(self):
        figure = on.rows(self.a, self.b, heights=[Px(240), Px(40)])
        self.assertEqual(figure.layout.children[0].size, Px(240))

    def test_cols__widths__should_size_the_columns(self):
        figure = on.cols(self.a, self.b, widths=[200, 100])
        widths = [cell["width"] for cell in self.compile(figure)["hconcat"]]
        self.assertEqual(widths, [200, 100])

    # ----------------------------------------------------------------- grid API

    def test_grid__full_rows__should_wrap_at_the_column_count(self):
        figure = on.grid([self.a, self.b, self.c, self.d], columns=2)
        self.assertEqual(figure.to_string(), "A B\nC D")

    def test_grid__partial_last_row__should_be_padded_with_spacers(self):
        figure = on.grid([self.a, self.b, self.c], columns=2)
        self.assertEqual(figure.to_string(), "A B\nC .")
        self.assertIs(figure.layout.children[1].children[1], Spacer(1))

    def test_grid__single_column__should_be_a_rows_group(self):
        figure = on.grid([self.a, self.b], columns=1)
        self.assertEqual(figure.to_string(), "A\nB")

    def test_grid__no_panel__should_raise(self):
        with self.assertRaises(ValueError):
            on.grid([], columns=2)

    def test_grid__non_positive_columns__should_raise(self):
        for columns in (0, -1, 1.5):
            with self.subTest(columns=columns):
                with self.assertRaises(ValueError):
                    on.grid([self.a, self.b], columns=columns)

    def test_grid__defaults__should_share_both_axes(self):
        figure = on.grid([self.a, self.b, self.c, self.d], columns=2)
        self.assertIs(figure.layout.share_x, True)
        self.assertIs(figure.layout.share_y, True)

    # -------------------------------------------------------------- sharing API

    def test_layout__default_sharing__should_follow_the_front_end(self):
        expected = {
            "rows": ("shared", "independent"),
            "cols": ("independent", "independent"),
            "layout": ("shared", "independent"),
            "grid": ("shared", "shared"),
        }
        figures = {
            "rows": on.rows(self.a, self.b),
            "cols": on.cols(self.a, self.b),
            "layout": on.layout("A\nB", A=self.a, B=self.b),
            "grid": on.grid([self.a, self.b, self.c, self.d], columns=2),
        }
        for kind, figure in figures.items():
            with self.subTest(kind=kind):
                scale = self.compile(figure)["resolve"]["scale"]
                self.assertEqual((scale["x"], scale["y"]), expected[kind])

    def test_grid__nested_row__should_inherit_the_sharing_of_the_grid(self):
        spec = self.compile(on.grid([self.a, self.b, self.c, self.d], columns=2))
        inner = spec["vconcat"][0]["resolve"]["scale"]
        self.assertEqual((inner["x"], inner["y"]), ("shared", "shared"))

    def test_layout__named_group__should_pin_a_union_domain(self):
        figure = on.layout(
            "A B\nC .",
            A=self.make_plot("a"),
            B=self.make_plot("b", 100),
            C=self.make_plot("c", -50),
            share_y="AC",
        )
        spec = self.compile(figure)
        pinned = self.encoding(spec["vconcat"][0]["hconcat"][0], "y")
        other = self.encoding(spec["vconcat"][0]["hconcat"][1], "y")
        member = self.encoding(spec["vconcat"][1]["hconcat"][0], "y")
        self.assertEqual(pinned["scale"]["domain"], [-50, 23])
        self.assertEqual(member["scale"]["domain"], [-50, 23])
        self.assertNotIn("scale", other)

    def test_layout__several_named_groups__should_pin_each_domain(self):
        figure = on.layout(
            "A B\nC D",
            A=self.make_plot("a"),
            B=self.make_plot("b", 100),
            C=self.make_plot("c"),
            D=self.make_plot("d", 100),
            share_y=["AC", "BD"],
        )
        spec = self.compile(figure)
        first = self.encoding(spec["vconcat"][0]["hconcat"][0], "y")
        second = self.encoding(spec["vconcat"][0]["hconcat"][1], "y")
        self.assertEqual(first["scale"]["domain"], [0, 23])
        self.assertEqual(second["scale"]["domain"], [100, 123])

    def test_layout__named_group_of_panel_objects__should_be_accepted(self):
        figure = on.layout(
            "A B", A=self.a, B=self.make_plot("b", 100), share_y=[[self.a]]
        )
        spec = self.compile(figure)
        self.assertIn("scale", self.encoding(spec["hconcat"][0], "y"))

    def test_layout__named_group_of_unknown_panel__should_raise(self):
        with self.assertRaises(ValueError) as raised:
            on.layout("A B", A=self.a, B=self.b, share_y="AZ")
        self.assertIn("Z", str(raised.exception))

    def test_layout__named_group_of_a_foreign_plot__should_raise(self):
        with self.assertRaises(ValueError):
            on.layout("A B", A=self.a, B=self.b, share_y=[[self.c]])

    def test_layout__named_group_of_one_character__should_be_allowed(self):
        figure = on.layout("A B", A=self.a, B=self.b, share_y="A")
        self.assertIn("scale", self.encoding(self.compile(figure)["hconcat"][0], "y"))

    # --------------------------------------------------------------- labels API

    def test_layout__labels_bottom__should_only_label_the_last_row(self):
        spec = self.compile(on.rows(self.a, self.b, self.c, labels="bottom"))
        hidden = ["axis" in self.encoding(row, "x") for row in spec["vconcat"]]
        self.assertEqual(hidden, [True, True, False])

    def test_layout__labels_all__should_label_every_row(self):
        spec = self.compile(on.rows(self.a, self.b, self.c, labels="all"))
        hidden = ["axis" in self.encoding(row, "x") for row in spec["vconcat"]]
        self.assertEqual(hidden, [False, False, False])

    def test_layout__unknown_labels__should_raise(self):
        with self.assertRaises(ValueError):
            on.layout("A\nB", A=self.a, B=self.b, labels="none")

    # --------------------------------------------------------------- to_string

    def test_to_string__every_front_end__should_round_trip(self):
        figures = [
            on.layout("A A B\nC C B", A=self.a, B=self.b, C=self.c),
            on.layout("A .\nA B", A=self.a, B=self.b),
            on.rows(self.a, self.b, self.c),
            on.cols(self.a, self.b),
            on.grid([self.a, self.b, self.c], columns=2),
            on.cols(on.rows(self.a, self.b), self.c),
        ]
        for figure in figures:
            with self.subTest(figure=figure.to_string()):
                spec = figure.to_string()
                rebuilt = on.layout(
                    spec,
                    {character: self.a for character in set(spec) - set(" \n.")},
                )
                self.assertEqual(rebuilt.to_string(), spec)

    def test_repr__figure__should_be_the_layout_string(self):
        figure = on.layout("A A B\nC C B", A=self.a, B=self.b, C=self.c)
        self.assertEqual(repr(figure), "A A B\nC C B")


if __name__ == "__main__":
    unittest.main()
