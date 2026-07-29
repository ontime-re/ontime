import unittest

from ontime.core.plotting._grid import parse, to_string
from ontime.core.plotting._layout import Cols, Panel, Rows, Spacer

ROUND_TRIPS = [
    "A",
    "A\nB",
    "A B\nC D",
    "A A B\nC C B",
    "T T\nA B",
    "A\nA\nB",
    "A .\nA B",
    "A A\nB C\nD D",
]


class TestParse(unittest.TestCase):
    """Test the layout string parser."""

    # ------------------------------------------------------------------ shapes

    def test_parse__two_by_two_grid__should_nest_rows_of_cols(self):
        self.assertIs(
            parse("A B\nC D"),
            Rows(
                (
                    Cols((Panel("A", 1), Panel("B", 1)), 1),
                    Cols((Panel("C", 1), Panel("D", 1)), 1),
                )
            ),
        )

    def test_parse__panel_spanning_two_rows__should_cut_vertically_first(self):
        self.assertIs(
            parse("A B\nC B"),
            Cols((Rows((Panel("A", 1), Panel("C", 1)), 1), Panel("B", 1))),
        )

    def test_parse__wide_column__should_carry_its_span_as_a_weight(self):
        self.assertIs(
            parse("A A B\nC C B"),
            Cols((Rows((Panel("A", 1), Panel("C", 1)), 2), Panel("B", 1))),
        )

    def test_parse__single_row__should_be_a_cols_group(self):
        self.assertIs(parse("A A B"), Cols((Panel("A", 2), Panel("B", 1))))

    def test_parse__single_character__should_be_a_panel(self):
        self.assertIs(parse("A"), Panel("A"))

    def test_parse__full_width_banner__should_be_a_rows_group(self):
        self.assertIs(
            parse("T T\nA B"),
            Rows((Panel("T", 1), Cols((Panel("A", 1), Panel("B", 1)), 1))),
        )

    def test_parse__gap_character__should_become_a_spacer(self):
        self.assertIs(
            parse("A .\nA B"),
            Cols((Panel("A", 1), Rows((Spacer(1), Panel("B", 1)), 1))),
        )

    def test_parse__whitespace__should_not_matter(self):
        self.assertIs(parse("AB\nCD"), parse("A B\nC D"))
        self.assertIs(parse("\n   A B\n   C D\n"), parse("A B\nC D"))
        self.assertIs(parse("A  B \n C  D"), parse("A B\nC D"))

    def test_parse__equal_layouts__should_be_the_same_object(self):
        self.assertIs(parse("A A B\nC C B"), parse("A A B\nC C B"))

    # ------------------------------------------------------------------ errors

    def test_parse__unsupported_character__should_raise_syntax_error(self):
        for spec in ("A-B", "[A][B]", "A / B", "A|B"):
            with self.subTest(spec=spec):
                with self.assertRaises(SyntaxError):
                    parse(spec)

    def test_parse__empty_string__should_raise_syntax_error(self):
        for spec in ("", "   ", "\n\n"):
            with self.subTest(spec=spec):
                with self.assertRaises(SyntaxError):
                    parse(spec)

    def test_parse__ragged_rows__should_raise_syntax_error(self):
        with self.assertRaises(SyntaxError):
            parse("A B\nC")

    def test_parse__non_rectangular_panel__should_raise_value_error(self):
        for spec in ("A B\nB A", "A A B\nC B B"):
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    parse(spec)

    def test_parse__interlocking_spans__should_raise_value_error(self):
        for spec in ("A A B\nC D B\nC E E", "A B B\nA C C\nD D C"):
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    parse(spec)

    def test_parse__not_a_string__should_raise_type_error(self):
        with self.assertRaises(TypeError):
            parse(["A", "B"])


class TestToString(unittest.TestCase):
    """Test the layout string printer."""

    def test_to_string__parsed_layout__should_round_trip(self):
        for spec in ROUND_TRIPS:
            with self.subTest(spec=spec):
                self.assertEqual(to_string(parse(spec)), spec)

    def test_to_string__printed_layout__should_be_idempotent(self):
        for spec in ROUND_TRIPS:
            with self.subTest(spec=spec):
                once = to_string(parse(spec))
                self.assertEqual(to_string(parse(once)), once)

    def test_to_string__reparsed_layout__should_give_the_same_tree(self):
        for spec in ROUND_TRIPS:
            with self.subTest(spec=spec):
                self.assertIs(parse(to_string(parse(spec))), parse(spec))

    def test_to_string__unnamed_panels__should_be_named_in_reading_order(self):
        self.assertEqual(to_string(Rows((Panel(None), Panel(None)))), "A\nB")
        self.assertEqual(to_string(Cols((Panel(None), Panel(None)))), "A B")

    def test_to_string__repeated_panel__should_be_named_once_per_cell(self):
        panel = Panel(None)
        self.assertEqual(to_string(Rows((panel, panel))), "A\nB")

    def test_to_string__sizes__should_not_appear(self):
        self.assertEqual(to_string(Rows((Panel("A", 3), Panel("B", 1)))), "A\nA\nA\nB")

    def test_to_string__spacer__should_be_a_gap(self):
        self.assertEqual(to_string(Cols((Panel("A", 1), Spacer(1)))), "A .")


if __name__ == "__main__":
    unittest.main()
