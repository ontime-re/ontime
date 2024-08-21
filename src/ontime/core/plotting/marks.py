from ..utils.dot_dict import DotDict
from ._marks.activity import activity
from ._marks.area import area
from ._marks.dots import dots
from ._marks.heatmap import heatmap
from ._marks.line import line
from ._marks.mark import mark

# Export all marks in a DotDict
marks = DotDict(
    {
        "activity": activity,
        "area": area,
        "dots": dots,
        "heatmap": heatmap,
        "line": line,
        "mark": mark,
    }
)
