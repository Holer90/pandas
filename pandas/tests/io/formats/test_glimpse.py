from io import StringIO
import textwrap

import numpy as np
import pytest

from pandas import (
    DataFrame,
    Series,
    option_context
)

from pandas.core.indexes.api import MultiIndex

@pytest.fixture
def duplicate_columns_frame() -> DataFrame:
    """Dataframe with duplicate column names."""
    return DataFrame(np.random.randn(1500, 4), columns=["a", "a", "b", "b"])


@pytest.fixture
def multiindex_columns_frame() -> DataFrame:
    """Dataframe with multiindex columns."""
    columns = MultiIndex(
        levels=[["foo", "bar", "baz"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        names=["first", "second"],
    )
    return DataFrame(
        np.arange(36).reshape(4, 9), index=["A", "B", "C", "D"], columns=columns
    )


@pytest.fixture
def pascal_series() -> Series:
    """Series with numbers from the Pascal Triangle Sequence"""
    data = [1, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 4, 6, 4, 1, 1, 5, 10, 10, 5, 1, 1, 6, 15, 20, 15, 6, 1, 1, 7, 21, 35, 35, 21, 7, 1, 1, 8, 28, 56, 70, 56, 28, 8, 1, 1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
    return Series(data, dtype=np.int64, name="pascal")


@pytest.fixture
def iris_frame() -> DataFrame:
    """DataFrame with the Iris dataset"""
    data = {
        "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7, 4.8, 5.4, 5.2, 5.5, 4.9, 5.0, 5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1, 4.8, 5.1, 4.6, 5.3, 5.0, 7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2, 5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4, 6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5, 5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3, 6.5, 7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6.0, 6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2, 7.4, 7.9, 6.4, 6.3, 6.1, 7.7, 6.3, 6.4, 6.0, 6.9, 6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9],
        "sepal_width": [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.6, 3.0, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.7, 3.3, 3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3.0, 2.8, 3.0, 2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3.0, 3.4, 3.1, 2.3, 3.0, 2.5, 2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8, 3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3.0, 2.5, 2.8, 3.2, 3.0, 3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3.0, 2.8, 3.0, 2.8, 3.8, 2.8, 2.8, 2.6, 3.0, 3.4, 3.1, 3.0, 3.1, 3.1, 3.1, 2.7, 3.2, 3.3, 3.0, 2.5, 3.0, 3.4, 3.0],
        "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6, 1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.4, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9, 1.4, 1.6, 1.4, 1.5, 1.4, 4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4.0, 4.9, 4.7, 4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4.0, 4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1, 6.0, 5.1, 5.9, 5.6, 5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9, 5.0, 5.7, 4.9, 6.7, 4.9, 5.7, 6.0, 4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1, 5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5.0, 5.2, 5.4, 5.1],
        "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2.0, 1.9, 2.1, 2.0, 2.4, 2.3, 1.8, 2.2, 2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6, 1.9, 2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0, 2.3, 1.8],
        "species": ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica']
    }
    return DataFrame(data)


def test_glimpse_empty():
    df = DataFrame()
    buf = StringIO()
    df.glimpse(buf=buf)
    result = buf.getvalue()
    expected = "DataFrame with 0 rows and 0 columns.\n"
    assert result == expected




@pytest.mark.parametrize(
    "fixture_func_name",
    [
        "int_frame",
        "float_frame",
        "datetime_frame",
        "duplicate_columns_frame",
    ],
)
def test_glimpse_smoke_test(fixture_func_name, request):
    """Tests if we print the expected number of lines."""
    frame = request.getfixturevalue(fixture_func_name)
    buf = StringIO()
    frame.glimpse(buf=buf, verbose=False)
    result = buf.getvalue().splitlines()
    assert len(result) == 5


@pytest.mark.parametrize(
    "fixture_func_name",
    [
        "int_frame",
        "float_frame",
        "datetime_frame",
        "duplicate_columns_frame",
    ],
)
def test_glimpse_verbose_smoke_test(fixture_func_name, request):
    """Tests if we print the expected number of lines to print in verbose-mode."""
    frame = request.getfixturevalue(fixture_func_name)
    buf = StringIO()
    frame.glimpse(buf=buf, verbose=True)
    result = buf.getvalue().splitlines()
    assert len(result) == 7



@pytest.mark.parametrize(
    "index, dtype, isna, notna, nunique, unique_values, emphasize, verbose",
    [
        (False, False, False, False, False, False, False, False),
        (True, False, False, False, False, False, False, True),
        (False, True, False, False, False, False, False, False),
        (False, False, True, False, False, False, False, True),
        (False, False, False, True, False, False, False, True),
        (False, False, False, False, True, False, False, True),
        (False, False, False, False, False, True, False, False),
        (False, False, False, False, False, False, True, False),
    ],
)
def test_glimpse_default_verbose_selection(index, dtype, isna, notna, nunique, unique_values, emphasize, verbose):
    """Tests if the default verbose selection is working as expected."""
    frame = DataFrame(np.random.randn(5, 8))
    io_default = StringIO()
    frame.glimpse(index=index,
                  dtype=dtype,
                  isna=isna,
                  notna=notna,
                  nunique=nunique,
                  unique_values=unique_values,
                  emphasize=emphasize,
                  buf=io_default)
    result = io_default.getvalue()

    io_explicit = StringIO()
    frame.glimpse(index=index,
                  dtype=dtype,
                  isna=isna,
                  notna=notna,
                  nunique=nunique,
                  unique_values=unique_values,
                  emphasize=emphasize,
                  buf=io_explicit,
                  verbose=verbose)
    expected = io_explicit.getvalue()

    assert result == expected


@pytest.mark.parametrize(
    "display_width, emphasize, verbose",
    [
        (80, False, False),
        (80, True, False),
        (80, False, True),
        (80, True, True),
        (120, False, False),
        (120, True, False),
        (120, False, True),
        (120, True, True),
    ],
)
def test_glimpse_width(display_width, emphasize, verbose):
    """
    Tests if we get the correct line width after trimming the values
    and adding escape characters.
    """
    frame = DataFrame(np.random.randn(20, 4))
    with option_context("display.width", display_width):
        buf = StringIO()
        frame.glimpse(buf=buf, emphasize=emphasize, verbose=verbose)
        lines = buf.getvalue().splitlines()
        result = [len(line) for line in lines[1:]]
        expected = list()
        if verbose is True:
            expected += [display_width] * 2

        if emphasize is True:
            expected += [display_width + 8] * 4
        else:
            expected += [display_width] * 4

        assert result == expected


def test_glimpse_duplicate_columns_shows_correct_dtypes():
    """Tests if we get the correct dtypes for columns with duplicate names."""
    io = StringIO()
    frame = DataFrame([[1, 2.0]], columns=["a", "a"])
    frame.glimpse(buf=io)
    lines = io.getvalue().splitlines(True)
    assert "a  \x1b[3m<int64>  \x1b[0m  1  \n" == lines[1]
    assert "a  \x1b[3m<float64>\x1b[0m  2.0\n" == lines[2]


def test_glimpse_shows_column_dtypes():
    """Tests if the different dtypes are correctly displayed."""
    dtypes = [
        "int64",
        "float64",
        "datetime64[ns]",
        "timedelta64[ns]",
        "complex128",
        "object",
        "bool",
    ]
    data = {}
    n = 10
    for i, dtype in enumerate(dtypes):
        data[i] = np.random.randint(2, size=n).astype(dtype)
    df = DataFrame(data)
    buf = StringIO()
    df.glimpse(buf=buf)
    result = buf.getvalue()
    print(result)
    for dtype in dtypes:
        assert f"<{dtype}>" in result


def test_glimpse_alignment_on_multiindex(request):
    """Tests if everything is correctly aligned when using a multiindex."""
    frame = request.getfixturevalue("multiindex_columns_frame")
    buf = StringIO()
    frame.glimpse(buf=buf)
    result = buf.getvalue()
    expected = textwrap.dedent(
        f"""\
    DataFrame with 4 rows and 9 columns.
    (foo, one)    \x1b[3m<int64>\x1b[0m  0, 9, 18, 27 
    (foo, two)    \x1b[3m<int64>\x1b[0m  1, 10, 19, 28
    (foo, three)  \x1b[3m<int64>\x1b[0m  2, 11, 20, 29
    (bar, one)    \x1b[3m<int64>\x1b[0m  3, 12, 21, 30
    (bar, two)    \x1b[3m<int64>\x1b[0m  4, 13, 22, 31
    (bar, three)  \x1b[3m<int64>\x1b[0m  5, 14, 23, 32
    (baz, one)    \x1b[3m<int64>\x1b[0m  6, 15, 24, 33
    (baz, two)    \x1b[3m<int64>\x1b[0m  7, 16, 25, 34
    (baz, three)  \x1b[3m<int64>\x1b[0m  8, 17, 26, 35
    """
    )

    assert result == expected

    buf_verbose = StringIO()
    frame.glimpse(buf=buf_verbose, verbose=True)
    result_verbose = buf_verbose.getvalue()
    expected_verbose = textwrap.dedent(
        f"""\
    DataFrame with 4 rows and 9 columns.
    Column        Dtype  Values       
    ------        -----  ------       
    (foo, one)    int64  0, 9, 18, 27 
    (foo, two)    int64  1, 10, 19, 28
    (foo, three)  int64  2, 11, 20, 29
    (bar, one)    int64  3, 12, 21, 30
    (bar, two)    int64  4, 13, 22, 31
    (bar, three)  int64  5, 14, 23, 32
    (baz, one)    int64  6, 15, 24, 33
    (baz, two)    int64  7, 16, 25, 34
    (baz, three)  int64  8, 17, 26, 35
    """
    )

    assert result_verbose == expected_verbose



@pytest.mark.parametrize(
    "verbose, fixture_name, expected",
    [
        (
            False,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            0  sepal_length  <float64>  5.1, 4.9, 4.7, 4.6, 5.0, 5.4 ...
            1  sepal_width   <float64>  3.5, 3.0, 3.2, 3.1, 3.6, 3.9 ...
            2  petal_length  <float64>  1.4, 1.4, 1.3, 1.5, 1.4, 1.7 ...
            3  petal_width   <float64>  0.2, 0.2, 0.2, 0.2, 0.2, 0.4 ...
            4  species       <object>   'setosa', 'setosa', 'setosa' ...
            """
        ),
        (
            True,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
             #   Column        Dtype    Values                          
            ---  ------        -----    ------                          
             0   sepal_length  float64  5.1, 4.9, 4.7, 4.6, 5.0, 5.4 ...
             1   sepal_width   float64  3.5, 3.0, 3.2, 3.1, 3.6, 3.9 ...
             2   petal_length  float64  1.4, 1.4, 1.3, 1.5, 1.4, 1.7 ...
             3   petal_width   float64  0.2, 0.2, 0.2, 0.2, 0.2, 0.4 ...
             4   species       object   'setosa', 'setosa', 'setosa' ...
            """
        ),
    ],
)
def test_glimpse_index_bool(verbose,
                            fixture_name,
                            expected,
                            request,
                            ):
    frame = request.getfixturevalue(fixture_name)
    buf = StringIO()
    frame.glimpse(buf=buf,
                  index=True,
                  verbose=verbose,
                  emphasize=False,
                  width=60)
    result = buf.getvalue()
    expected_dedent = textwrap.dedent(expected)
    assert result == expected_dedent


@pytest.mark.parametrize(
    "verbose, fixture_name, expected",
    [
        (
            False,
            "pascal_series",
            """\
            Series (pascal) with 55 rows.
            pascal  <int64>  1, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 4, 6,  ...
            """
        ),
        (
            False,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            sepal_length  <float64>  5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4 ...
            sepal_width   <float64>  3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3 ...
            petal_length  <float64>  1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1 ...
            petal_width   <float64>  0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0 ...
            species       <object>   'setosa', 'setosa', 'setosa', ' ...
            """
        ),
        (
            True,
            "pascal_series",
            """\
            Series (pascal) with 55 rows.
            Name    Dtype  Values                                       
            ----    -----  ------                                       
            pascal  int64  1, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1, 4, 6, 4, ...
            """
        ),
        (
            True,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            Column        Dtype    Values                               
            ------        -----    ------                               
            sepal_length  float64  5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6 ...
            sepal_width   float64  3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4 ...
            petal_length  float64  1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4 ...
            petal_width   float64  0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3 ...
            species       object   'setosa', 'setosa', 'setosa', 'se ...
            """
        ),
    ],
)
def test_glimpse_dtype_bool(verbose,
                            fixture_name,
                            expected,
                            request,
                            ):
    frame = request.getfixturevalue(fixture_name)
    buf = StringIO()
    frame.glimpse(buf=buf,
                  dtype=True,
                  verbose=verbose,
                  emphasize=False,
                  width=60)
    result = buf.getvalue()
    expected_dedent = textwrap.dedent(expected)
    assert result == expected_dedent


@pytest.mark.parametrize(
    "verbose, fixture_name, expected",
    [
        (
            False,
            "pascal_series",
            """\
            Series (pascal) with 55 rows.
            pascal  <int64>  (0/55)  1, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1 ...
            """
        ),
        (
            False,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            sepal_length  <float64>  (0/150)  5.1, 4.9, 4.7, 4.6, 5. ...
            sepal_width   <float64>  (0/150)  3.5, 3.0, 3.2, 3.1, 3. ...
            petal_length  <float64>  (0/150)  1.4, 1.4, 1.3, 1.5, 1. ...
            petal_width   <float64>  (0/150)  0.2, 0.2, 0.2, 0.2, 0. ...
            species       <object>   (0/150)  'setosa', 'setosa', 's ...
            """
        ),
        (
            True,
            "pascal_series",
            """\
            Series (pascal) with 55 rows.
            Name    Dtype  Null    Values                               
            ----    -----  ----    ------                               
            pascal  int64  0 null  1, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1,  ...
            """
        ),
        (
            True,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            Column        Dtype    Null    Values                       
            ------        -----    ----    ------                       
            sepal_length  float64  0 null  5.1, 4.9, 4.7, 4.6, 5.0,  ...
            sepal_width   float64  0 null  3.5, 3.0, 3.2, 3.1, 3.6,  ...
            petal_length  float64  0 null  1.4, 1.4, 1.3, 1.5, 1.4,  ...
            petal_width   float64  0 null  0.2, 0.2, 0.2, 0.2, 0.2,  ...
            species       object   0 null  'setosa', 'setosa', 'seto ...
            """
        ),
    ],
)
def test_glimpse_isna_bool(verbose,
                           fixture_name,
                           expected,
                           request,
                           ):
    frame = request.getfixturevalue(fixture_name)
    buf = StringIO()
    frame.glimpse(buf=buf,
                  isna=True,
                  verbose=verbose,
                  emphasize=False,
                  width=60)
    result = buf.getvalue()
    expected_dedent = textwrap.dedent(expected)
    assert result == expected_dedent


@pytest.mark.parametrize(
    "verbose, fixture_name, expected",
    [
        (
            False,
            "pascal_series",
            """\
            Series (pascal) with 55 rows.
            pascal  <int64>  (0/55)  1, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1 ...
            """
        ),
        (
            False,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            sepal_length  <float64>  (0/150)  5.1, 4.9, 4.7, 4.6, 5. ...
            sepal_width   <float64>  (0/150)  3.5, 3.0, 3.2, 3.1, 3. ...
            petal_length  <float64>  (0/150)  1.4, 1.4, 1.3, 1.5, 1. ...
            petal_width   <float64>  (0/150)  0.2, 0.2, 0.2, 0.2, 0. ...
            species       <object>   (0/150)  'setosa', 'setosa', 's ...
            """
        ),
        (
            True,
            "pascal_series",
            """\
            Series (pascal) with 55 rows.
            Name    Dtype  Non-null     Values                          
            ----    -----  --------     ------                          
            pascal  int64  55 non-null  1, 1, 1, 1, 2, 1, 1, 3, 3, 1 ...
            """
        ),
        (
            True,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            Column        Dtype    Non-null      Values                 
            ------        -----    --------      ------                 
            sepal_length  float64  150 non-null  5.1, 4.9, 4.7, 4.6, ...
            sepal_width   float64  150 non-null  3.5, 3.0, 3.2, 3.1, ...
            petal_length  float64  150 non-null  1.4, 1.4, 1.3, 1.5, ...
            petal_width   float64  150 non-null  0.2, 0.2, 0.2, 0.2, ...
            species       object   150 non-null  'setosa', 'setosa', ...
            """
        ),
    ],
)
def test_glimpse_notna_bool(verbose,
                            fixture_name,
                            expected,
                            request,
                            ):
    frame = request.getfixturevalue(fixture_name)
    buf = StringIO()
    frame.glimpse(buf=buf,
                  notna=True,
                  verbose=verbose,
                  emphasize=False,
                  width=60)
    result = buf.getvalue()
    expected_dedent = textwrap.dedent(expected)
    assert result == expected_dedent


@pytest.mark.parametrize(
    "verbose, fixture_name, expected",
    [
        (
            False,
            "pascal_series",
            """\
            Series (pascal) with 55 rows.
            pascal  <int64>  |20|  1, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1,  ...
            """
        ),
        (
            False,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            sepal_length  <float64>  |35|  5.1, 4.9, 4.7, 4.6, 5.0,  ...
            sepal_width   <float64>  |23|  3.5, 3.0, 3.2, 3.1, 3.6,  ...
            petal_length  <float64>  |43|  1.4, 1.4, 1.3, 1.5, 1.4,  ...
            petal_width   <float64>  |22|  0.2, 0.2, 0.2, 0.2, 0.2,  ...
            species       <object>   |3|   'setosa', 'setosa', 'seto ...
            """
        ),
        (
            True,
            "pascal_series",
            """\
            Series (pascal) with 55 rows.
            Name    Dtype  N-unique   Values                            
            ----    -----  --------   ------                            
            pascal  int64  20 unique  1, 1, 1, 1, 2, 1, 1, 3, 3, 1,  ...
            """
        ),
        (
            True,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            Column        Dtype    N-unique   Values                    
            ------        -----    --------   ------                    
            sepal_length  float64  35 unique  5.1, 4.9, 4.7, 4.6, 5. ...
            sepal_width   float64  23 unique  3.5, 3.0, 3.2, 3.1, 3. ...
            petal_length  float64  43 unique  1.4, 1.4, 1.3, 1.5, 1. ...
            petal_width   float64  22 unique  0.2, 0.2, 0.2, 0.2, 0. ...
            species       object   3 unique   'setosa', 'setosa', 's ...
            """
        ),
    ],
)
def test_glimpse_nunique_bool(verbose,
                              fixture_name,
                              expected,
                              request,
                              ):
    frame = request.getfixturevalue(fixture_name)
    buf = StringIO()
    frame.glimpse(buf=buf,
                  nunique=True,
                  verbose=verbose,
                  emphasize=False,
                  width=60)
    result = buf.getvalue()
    expected_dedent = textwrap.dedent(expected)
    assert result == expected_dedent


@pytest.mark.parametrize(
    "verbose, fixture_name, expected",
    [
        (
            False,
            "pascal_series",
            """\
            Series (pascal) with 55 rows.
            pascal  <int64>  1, 2, 3, 4, 6, 5, 10, 15, 20, 7, 21, 35 ...
            """
        ),
        (
            False,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            sepal_length  <float64>  5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4 ...
            sepal_width   <float64>  3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3 ...
            petal_length  <float64>  1.4, 1.3, 1.5, 1.7, 1.6, 1.1, 1 ...
            petal_width   <float64>  0.2, 0.4, 0.3, 0.1, 0.5, 0.6, 1 ...
            species       <object>   'setosa', 'versicolor', 'virgin ...
            """
        ),
        (
            True,
            "pascal_series",
            """\
            Series (pascal) with 55 rows.
            Name    Dtype  Unique values                                
            ----    -----  -------------                                
            pascal  int64  1, 2, 3, 4, 6, 5, 10, 15, 20, 7, 21, 35,  ...
            """
        ),
        (
            True,
            "iris_frame",
            """\
            DataFrame with 150 rows and 5 columns.
            Column        Dtype    Unique values                        
            ------        -----    -------------                        
            sepal_length  float64  5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.4 ...
            sepal_width   float64  3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4 ...
            petal_length  float64  1.4, 1.3, 1.5, 1.7, 1.6, 1.1, 1.2 ...
            petal_width   float64  0.2, 0.4, 0.3, 0.1, 0.5, 0.6, 1.4 ...
            species       object   'setosa', 'versicolor', 'virginic ...
            """
        ),
    ],
)
def test_glimpse_unique_values_bool(verbose,
                                    fixture_name,
                                    expected,
                                    request,
                                    ):
    frame = request.getfixturevalue(fixture_name)
    buf = StringIO()
    frame.glimpse(buf=buf,
                  unique_values=True,
                  verbose=verbose,
                  emphasize=False,
                  width=60)
    result = buf.getvalue()
    expected_dedent = textwrap.dedent(expected)
    assert result == expected_dedent


def test_glimpse_multiple_arguments(request):
    frame = request.getfixturevalue("iris_frame")
    buf = StringIO()
    frame.glimpse(buf=buf,
                  index=True,
                  dtype=False,
                  nunique=True,
                  unique_values=True,
                  width=70,
                  emphasize=False)
    result = buf.getvalue()
    expected = textwrap.dedent(
        """\
        DataFrame with 150 rows and 5 columns.
         #   Column        N-unique   Unique values                           
        ---  ------        --------   -------------                           
         0   sepal_length  35 unique  5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.4, 4 ...
         1   sepal_width   23 unique  3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 2 ...
         2   petal_length  43 unique  1.4, 1.3, 1.5, 1.7, 1.6, 1.1, 1.2, 1 ...
         3   petal_width   22 unique  0.2, 0.4, 0.3, 0.1, 0.5, 0.6, 1.4, 1 ...
         4   species       3 unique   'setosa', 'versicolor', 'virginica'     
        """
    )
    assert result == expected


def test_glimpse_emphasize(request):
    frame = request.getfixturevalue("iris_frame")
    buf = StringIO()
    frame.glimpse(buf=buf,
                  index=True,
                  dtype=True,
                  nunique=True,
                  isna=True,
                  notna=True,
                  verbose=False,
                  width=65,
                  emphasize=True)
    result = buf.getvalue()
    expected = textwrap.dedent(
        """\
        DataFrame with 150 rows and 5 columns.
        0  sepal_length  \x1b[3m<float64>  (0/150)  |35|\x1b[0m  5.1, 4.9, 4.7, 4.6 ...
        1  sepal_width   \x1b[3m<float64>  (0/150)  |23|\x1b[0m  3.5, 3.0, 3.2, 3.1 ...
        2  petal_length  \x1b[3m<float64>  (0/150)  |43|\x1b[0m  1.4, 1.4, 1.3, 1.5 ...
        3  petal_width   \x1b[3m<float64>  (0/150)  |22|\x1b[0m  0.2, 0.2, 0.2, 0.2 ...
        4  species       \x1b[3m<object>   (0/150)  |3| \x1b[0m  'setosa', 'setosa' ...
        """
    )
    assert result == expected


def test_glimpse_emphasize_edge_case(request):
    frame = request.getfixturevalue("iris_frame")
    buf = StringIO()
    frame.glimpse(buf=buf,
                  index=True,
                  dtype=True,
                  nunique=False,
                  isna=True,
                  notna=True,
                  verbose=True,
                  width=65,
                  emphasize=True)
    result = buf.getvalue()
    expected = textwrap.dedent(
        """\
        DataFrame with 150 rows and 5 columns.
         #   Column        Dtype    Null    Non-null      Values         
        ---  ------        -----    ----    --------      ------         
         0   sepal_length  \x1b[3mfloat64  0 null  150 non-null\x1b[0m  5.1, 4.9, 4 ...
         1   sepal_width   \x1b[3mfloat64  0 null  150 non-null\x1b[0m  3.5, 3.0, 3 ...
         2   petal_length  \x1b[3mfloat64  0 null  150 non-null\x1b[0m  1.4, 1.4, 1 ...
         3   petal_width   \x1b[3mfloat64  0 null  150 non-null\x1b[0m  0.2, 0.2, 0 ...
         4   species       \x1b[3mobject   0 null  150 non-null\x1b[0m  'setosa', ' ...
        """
    )
    assert result == expected






