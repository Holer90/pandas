from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import sys
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)

from pandas._config import get_option

from pandas._typing import (
    Dtype,
    WriteBuffer,
)

from pandas.core.indexes.api import Index

from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing

# todo: is there a way to fix circular import? maybe from pandas.core.series import Series??
#from pandas.core.frame import Series


if TYPE_CHECKING:
    from pandas.core.frame import (
        DataFrame,
        Series,
    )


frame_max_cols_sub = dedent(
    """\
    max_cols : int, optional
        When to switch from the verbose to the truncated output. If the
        DataFrame has more than `max_cols` columns, the truncated output
        is used. By default, the setting in
        ``pandas.options.display.max_info_columns`` is used."""
)


show_counts_sub = dedent(
    """\
    show_counts : bool, optional
        Whether to show the non-null counts. By default, this is shown
        only if the DataFrame is smaller than
        ``pandas.options.display.max_info_rows`` and
        ``pandas.options.display.max_info_columns``. A value of True always
        shows the counts, and False never shows the counts."""
)

null_counts_sub = dedent(
    """
    null_counts : bool, optional
        .. deprecated:: 1.2.0
            Use show_counts instead."""
)


frame_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    >>> df = pd.DataFrame({"int_col": int_values, "text_col": text_values,
    ...                   "float_col": float_values})
    >>> df
        int_col text_col  float_col
    0        1    alpha       0.00
    1        2     beta       0.25
    2        3    gamma       0.50
    3        4    delta       0.75
    4        5  epsilon       1.00

    Prints information of all columns:

    >>> df.info(verbose=True)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
     #   Column     Non-Null Count  Dtype
    ---  ------     --------------  -----
     0   int_col    5 non-null      int64
     1   text_col   5 non-null      object
     2   float_col  5 non-null      float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 248.0+ bytes

    Prints a summary of columns count and its dtypes but not per column
    information:

    >>> df.info(verbose=False)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Columns: 3 entries, int_col to float_col
    dtypes: float64(1), int64(1), object(1)
    memory usage: 248.0+ bytes

    Pipe output of DataFrame.info to buffer instead of sys.stdout, get
    buffer content and writes to a text file:

    >>> import io
    >>> buffer = io.StringIO()
    >>> df.info(buf=buffer)
    >>> s = buffer.getvalue()
    >>> with open("df_info.txt", "w",
    ...           encoding="utf-8") as f:  # doctest: +SKIP
    ...     f.write(s)
    260

    The `memory_usage` parameter allows deep introspection mode, specially
    useful for big DataFrames and fine-tune memory optimization:

    >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
    >>> df = pd.DataFrame({
    ...     'column_1': np.random.choice(['a', 'b', 'c'], 10 ** 6),
    ...     'column_2': np.random.choice(['a', 'b', 'c'], 10 ** 6),
    ...     'column_3': np.random.choice(['a', 'b', 'c'], 10 ** 6)
    ... })
    >>> df.info()
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 3 columns):
     #   Column    Non-Null Count    Dtype
    ---  ------    --------------    -----
     0   column_1  1000000 non-null  object
     1   column_2  1000000 non-null  object
     2   column_3  1000000 non-null  object
    dtypes: object(3)
    memory usage: 22.9+ MB

    >>> df.info(memory_usage='deep')
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000000 entries, 0 to 999999
    Data columns (total 3 columns):
     #   Column    Non-Null Count    Dtype
    ---  ------    --------------    -----
     0   column_1  1000000 non-null  object
     1   column_2  1000000 non-null  object
     2   column_3  1000000 non-null  object
    dtypes: object(3)
    memory usage: 165.9 MB"""
)


frame_see_also_sub = dedent(
    """\
    DataFrame.describe: Generate descriptive statistics of DataFrame
        columns.
    DataFrame.memory_usage: Memory usage of DataFrame columns."""
)


frame_sub_kwargs_glimpse = {
    "klass": "DataFrame",
    "type_sub": " and columns",
    "max_cols_sub": frame_max_cols_sub,
    "show_counts_sub": show_counts_sub,
    "null_counts_sub": null_counts_sub,
    "examples_sub": frame_examples_sub,
    "see_also_sub": frame_see_also_sub,
    "version_added_sub": "",
}


series_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
    >>> s = pd.Series(text_values, index=int_values)
    >>> s.info()
    <class 'pandas.core.series.Series'>
    Int64Index: 5 entries, 1 to 5
    Series name: None
    Non-Null Count  Dtype
    --------------  -----
    5 non-null      object
    dtypes: object(1)
    memory usage: 80.0+ bytes

    Prints a summary excluding information about its values:

    >>> s.info(verbose=False)
    <class 'pandas.core.series.Series'>
    Int64Index: 5 entries, 1 to 5
    dtypes: object(1)
    memory usage: 80.0+ bytes

    Pipe output of Series.info to buffer instead of sys.stdout, get
    buffer content and writes to a text file:

    >>> import io
    >>> buffer = io.StringIO()
    >>> s.info(buf=buffer)
    >>> s = buffer.getvalue()
    >>> with open("df_info.txt", "w",
    ...           encoding="utf-8") as f:  # doctest: +SKIP
    ...     f.write(s)
    260

    The `memory_usage` parameter allows deep introspection mode, specially
    useful for big Series and fine-tune memory optimization:

    >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
    >>> s = pd.Series(np.random.choice(['a', 'b', 'c'], 10 ** 6))
    >>> s.info()
    <class 'pandas.core.series.Series'>
    RangeIndex: 1000000 entries, 0 to 999999
    Series name: None
    Non-Null Count    Dtype
    --------------    -----
    1000000 non-null  object
    dtypes: object(1)
    memory usage: 7.6+ MB

    >>> s.info(memory_usage='deep')
    <class 'pandas.core.series.Series'>
    RangeIndex: 1000000 entries, 0 to 999999
    Series name: None
    Non-Null Count    Dtype
    --------------    -----
    1000000 non-null  object
    dtypes: object(1)
    memory usage: 55.3 MB"""
)


series_see_also_sub = dedent(
    """\
    Series.describe: Generate descriptive statistics of Series.
    Series.memory_usage: Memory usage of Series."""
)


series_sub_kwargs_glimpse = {
    "klass": "Series",
    "type_sub": "",
    "max_cols_sub": "",
    "show_counts_sub": show_counts_sub,
    "null_counts_sub": "",
    "examples_sub": series_examples_sub,
    "see_also_sub": series_see_also_sub,
    "version_added_sub": "\n.. versionadded:: 1.4.0\n",
}


GLIMPSE_DOCSTRING = dedent(
    """
    Print a concise summary of a {klass}.

    This method prints information about a {klass} including
    the index dtype{type_sub}, non-null values and memory usage.
    {version_added_sub}\

    Parameters
    ----------
    verbose : bool, optional
        Whether to print the full summary. By default, the setting in
        ``pandas.options.display.max_info_columns`` is followed.
    buf : writable buffer, defaults to sys.stdout
        Where to send the output. By default, the output is printed to
        sys.stdout. Pass a writable buffer if you need to further process
        the output.\
    {max_cols_sub}
    memory_usage : bool, str, optional
        Specifies whether total memory usage of the {klass}
        elements (including the index) should be displayed. By default,
        this follows the ``pandas.options.display.memory_usage`` setting.

        True always show memory usage. False never shows memory usage.
        A value of 'deep' is equivalent to "True with deep introspection".
        Memory usage is shown in human-readable units (base-2
        representation). Without deep introspection a memory estimation is
        made based in column dtype and number of rows assuming values
        consume the same memory amount for corresponding dtypes. With deep
        memory introspection, a real memory usage calculation is performed
        at the cost of computational resources. See the
        :ref:`Frequently Asked Questions <df-memory-usage>` for more
        details.
    {show_counts_sub}{null_counts_sub}

    Returns
    -------
    None
        This method prints a summary of a {klass} and returns None.

    See Also
    --------
    {see_also_sub}

    Examples
    --------
    {examples_sub}
    """
)


def _put_str(s: str | Dtype, space: int) -> str:
    """
    Make string of specified length, padding to the right if necessary.

    Parameters
    ----------
    s : Union[str, Dtype]
        String to be formatted.
    space : int
        Length to force string to be of.

    Returns
    -------
    str
        String coerced to given length.

    Examples
    --------
    >>> pd.io.formats.info._put_str("panda", 6)
    'panda '
    >>> pd.io.formats.info._put_str("panda", 4)
    'pand'
    """
    return str(s)[:space].ljust(space)

def _trim_str(s: str, length: int) -> str:
    """
    Crop a string from the right and adding '...' if its too long (ignoring spaces)

    Parameters
    ----------
    s : str
        String to be formatted.
    length : int
        Length to force string to be of.

    Returns
    -------
    str
        String of length no longer than desired length

    Examples
    --------
    >>> pd.io.formats.info._trim_str("red pandas", 10)
    'red pandas'
    >>> pd.io.formats.info._trim_str("red pandas", 8)
    'red pa ...'
    >>> pd.io.formats.info._trim_str("pandas    ", 8)
    'pandas    '
    """
    if len(str(s)) > length:
        if s[length-4:length] == "    ":
            return str(s[:length])
        else:
            return f"{s[:length - 4]} ..."
    else:
        return str(s)


class BaseGlimpseInfo(ABC):
    """
    Base class for DataFrameGlimpseInfo and SeriesGlimpseInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either dataframe or series.
    """

    data: DataFrame | Series

    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """

    @property
    @abstractmethod
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""

    @property
    @abstractmethod
    def null_counts(self) -> Sequence[int]:
        """Sequence of null counts for all columns or column (if series)."""

    @property
    @abstractmethod
    def nunique_counts(self) -> Sequence[int]:
        """Sequence of counts of unique element for all columns or column (if series)."""

    @property
    @abstractmethod
    def value_strings(self) -> Sequence[str]:
        """Sequence of string representing the unique values."""

    @property
    @abstractmethod
    def unique_value_strings(self) -> Sequence[str]:
        """Sequence of string representing the values."""

    @abstractmethod
    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        dtype: bool | None,
        notna: bool | None,
        isna: bool | None,
        nunique: bool | None,
        unique: bool | None,
    ) -> None:
        pass


class DataFrameGlimpseInfo(BaseGlimpseInfo):
    """
    Class storing dataframe-specific info needed to glimpse.
    """

    def __init__(
        self,
        data: DataFrame,
    ) -> None:
        self.data: DataFrame = data

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the DataFrame's columns.
        """
        return self.data.dtypes

    @property
    def ids(self) -> Index:
        """
        Column names.

        Returns
        -------
        ids : Index
            DataFrame's column names.
        """
        return self.data.columns

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return len(self.ids)

    @property
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""
        return self.data.count()

    @property
    def null_counts(self) -> Sequence[int]:
        """Sequence of null counts for all columns or column (if series)."""
        return self.data.isna().sum()

    @property
    def nunique_counts(self) -> Sequence[int]:
        """Sequence of counts of unique elements for all columns or column (if series)."""
        return _get_nunique_without_unhashable_error(self.data)

    @property
    def value_strings(self) -> Sequence[str]:
        # Calculate the (worst case) number of elements needed to be included in the values string.
        # Note: As the 'Column'-column takes up 7 characters and each element takes up at least 3 characters we get.

        # imported here to avoid circular imports from partially imported module.
        from pandas import Series

        display_width = get_option("display.width")
        value_strings_max_width = min(display_width - 7, 300)   # todo: is 300 a good value here? (to avoid printing too much?)
        number_of_elements_to_include = int(1 + value_strings_max_width / 3)

        s = Series(dtype=str)
        for col in self.ids:
            s[col] = ', '.join(map(
                lambda x: pprint_thing(x, quote_strings=True),
                self.data[col].head(number_of_elements_to_include).to_list()
            ))[:value_strings_max_width]
        return s

    @property
    def unique_value_strings(self) -> Sequence[str]:
        # imported here to avoid circular imports from partially imported module.
        from pandas import Series

        display_width = get_option("display.width")
        value_strings_max_width = min(display_width - 7, 300)   # todo: should we fetch 300 from somewhere?

        s = Series(dtype=str)
        for col in self.ids:
            s[col] = ', '.join(map(
                lambda x: pprint_thing(x, quote_strings=True),
                self.data[col].unique()
            ))[:value_strings_max_width]
        return s

    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        dtype: bool | None,
        notna: bool | None,
        isna: bool | None,
        nunique: bool | None,
        unique: bool | None,
    ) -> None:
        printer = DataFrameGlimpsePrinter(
            info=self,
            include_dtype=dtype,
            include_non_null_count=notna,
            include_null_count=isna,
            include_nunique=nunique,
            unique_values=unique
        )
        printer.to_buffer(buf)


class SeriesGlimpseInfo(BaseGlimpseInfo):
    """
    Class storing series-specific info.
    """

    def __init__(
        self,
        data: Series,
    ) -> None:
        self.data: Series = data

    def render(
            self,
            *,
            buf: WriteBuffer[str] | None,
            dtype: bool | None,
            notna: bool | None,
            isna: bool | None,
            nunique: bool | None,
            unique: bool | None,
    ) -> None:
        printer = SeriesGlimpsePrinter(
            info=self,
            include_dtype=dtype,
            include_non_null_count=notna,
            include_null_count=isna,
            include_nunique=nunique,
            unique_values=unique
        )
        printer.to_buffer(buf)

    @property
    def dtypes(self) -> Iterable[Dtype]:
        return [self.data.dtypes]

    @property
    def non_null_counts(self) -> Sequence[int]:
        return [self.data.count()]

    @property
    def null_counts(self) -> Sequence[int]:
        return [self.data.isna().sum()]

    @property
    def nunique_counts(self) -> Sequence[int]:
        # todo: this looks shady - reconsider??
        try:
            return [self.data.nunique()]
        except:
            return ['unhashable']
        #return [self.data.nunique()]

    @property
    def value_strings(self) -> Sequence[str]:
        # Calculate the (worst case) number of elements needed to be included in the values string.
        # Note: As the 'Column'-column takes up 7 characters and each element takes up at least 3 characters we get.


        # imported here to avoid circular imports from partially imported module.
        from pandas import Series

        display_width = get_option("display.width")
        value_strings_max_width = min(display_width - 7, 300)  # todo: is 300 a good value here? (to avoid printing too much?)
        number_of_elements_to_include = int(1 + value_strings_max_width / 3)

        # todo: this feels like a terrible way to do it..
        s = Series(dtype=str)
        s[self.data.name] = ', '.join(map(
                lambda x: pprint_thing(x, quote_strings=True),
                self.data.head(number_of_elements_to_include).to_list()
            ))[:value_strings_max_width]
        return s

    @property
    def unique_value_strings(self) -> Sequence[str]:
        from pandas import Series
        display_width = get_option("display.width")
        value_strings_max_width = min(display_width - 7, 300)  # todo: should we fetch 300 from somewhere?
        s = Series(dtype=str)
        s[self.data.name] = ', '.join(map(
                lambda x: pprint_thing(x, quote_strings=True),
                self.data.unique()
            ))[:value_strings_max_width]
        return s



class GlimpsePrinterAbstract:
    """
    Class for printing dataframe or series glimpse.
    """

    def to_buffer(self, buf: WriteBuffer[str] | None = None) -> None:
        """Save dataframe info into buffer."""
        table_builder = self._create_table_builder()
        lines = table_builder.get_lines()
        if buf is None:  # pragma: no cover
            buf = sys.stdout
        fmt.buffer_put_lines(buf, lines)

    @abstractmethod
    def _create_table_builder(self) -> TableBuilderAbstract:
        """Create instance of table builder."""


class DataFrameGlimpsePrinter(GlimpsePrinterAbstract):
    """
    Class for printing dataframe glimpse.

    Parameters
    ----------
    info : DataFrameGlimpseInfo
        Instance of DataFrameGlimpseInfo.
    include_dtype: bool, optional
        Whether to show the dtypes.
    include_non_null_count: bool, optional
        Whether to show the non-null counts.
    include_null_count: bool, optional
        Whether to show the null counts.
    include_nunique: bool, optional
        Whether to show the number of unique values.
    unique_values: bool, optional
        Whether to show the unique values.
    """

    def __init__(
        self,
        info: DataFrameGlimpseInfo,
        include_dtype: bool | None = None,
        include_non_null_count: bool | None = None,
        include_null_count: bool | None = None,
        include_nunique: bool | None = None,
        unique_values: bool | None = None,
    ) -> None:
        self.info = info
        self.data = info.data
        self.include_dtype = self._initialize_dtype(include_dtype)
        self.include_non_null_count = self._initialize_non_null_count(include_non_null_count)
        self.include_null_count = self._initialize_null_count(include_null_count)
        self.include_nunique = self._initialize_nunique(include_nunique)
        self.unique_values = self._initialize_unique_values(unique_values)

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return self.info.col_count

    def _initialize_dtype(self, include_dtype: bool | None) -> bool:
        if include_dtype is None:
            return True
        else:
            return include_dtype

    def _initialize_non_null_count(self, include_non_null_count: bool | None) -> bool:
        if include_non_null_count is None:
            return False
        else:
            return include_non_null_count

    def _initialize_null_count(self, include_null_count: bool | None) -> bool:
        if include_null_count is None:
            return False
        else:
            return include_null_count

    def _initialize_nunique(self, include_nunique: bool | None) -> bool:
        if include_nunique is None:
            return False
        else:
            return include_nunique

    def _initialize_unique_values(self, unique_values: bool | None) -> bool:
        if unique_values is None:
            return False
        else:
            return unique_values

    def _create_table_builder(self) -> DataFrameTableBuilder:
        """
        Create instance of table builder based on desired columns in the glimpse.
        """
        return DataFrameTableBuilder(
            info=self.info,
            include_dtype=self.include_dtype,
            include_non_null_count=self.include_non_null_count,
            include_null_count=self.include_null_count,
            include_nunique=self.include_nunique,
            unique_values=self.unique_values,
        )


class SeriesGlimpsePrinter(GlimpsePrinterAbstract):
    """Class for printing series info.

    Parameters
    ----------
    info : SeriesGlimpseInfo
        Instance of SeriesGlimpseInfo.
    include_dtype: bool, optional
        Whether to show the dtypes.
    include_non_null_count: bool, optional
        Whether to show the non-null counts.
    include_null_count: bool, optional
        Whether to show the null counts.
    include_nunique: bool, optional
        Whether to show the number of unique values.
    unique_values: bool, optional
        Whether to show the unique values.
    """

    def __init__(
        self,
        info: SeriesGlimpseInfo,
        include_dtype: bool | None = None,
        include_non_null_count: bool | None = None,
        include_null_count: bool | None = None,
        include_nunique: bool | None = None,
        unique_values: bool | None = None,
    ) -> None:
        self.info = info
        self.data = info.data
        self.include_dtype = self._initialize_dtype(include_dtype)
        self.include_non_null_count = self._initialize_non_null_count(include_non_null_count)
        self.include_null_count = self._initialize_null_count(include_null_count)
        self.include_nunique = self._initialize_nunique(include_nunique)
        self.unique_values = self._initialize_unique_values(unique_values)

    def _initialize_dtype(self, include_dtype: bool | None) -> bool:
        if include_dtype is None:
            return True
        else:
            return include_dtype

    def _initialize_non_null_count(self, include_non_null_count: bool | None) -> bool:
        if include_non_null_count is None:
            return False
        else:
            return include_non_null_count

    def _initialize_null_count(self, include_null_count: bool | None) -> bool:
        if include_null_count is None:
            return False
        else:
            return include_null_count

    def _initialize_nunique(self, include_nunique: bool | None) -> bool:
        if include_nunique is None:
            return False
        else:
            return include_nunique

    def _initialize_unique_values(self, unique_values: bool | None) -> bool:
        if unique_values is None:
            return False
        else:
            return unique_values

    def _create_table_builder(self) -> SeriesTableBuilder:
        """
        Create instance of table builder based on desired columns in the glimpse.
        """
        return SeriesTableBuilder(
            info=self.info,
            include_dtype=self.include_dtype,
            include_non_null_count=self.include_non_null_count,
            include_null_count=self.include_null_count,
            include_nunique=self.include_nunique,
            unique_values=self.unique_values,
        )


class TableBuilderAbstract(ABC):
    """
    Abstract builder for info table.
    """

    _lines: list[str]
    info: BaseGlimpseInfo

    @abstractmethod
    def get_lines(self) -> list[str]:
        """Product in a form of list of lines (strings)."""

    @property
    def data(self) -> DataFrame | Series:
        return self.info.data

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """Dtypes of each of the DataFrame's columns."""
        return self.info.dtypes

    @property
    def non_null_counts(self) -> Sequence[int]:
        return self.info.non_null_counts

    @property
    def null_counts(self) -> Sequence[int]:
        return self.info.null_counts

    @property
    def nunique_counts(self) -> Sequence[int]:
        return self.info.nunique_counts

    @property
    def value_strings(self) -> Iterable[str]:
        return self.info.value_strings

    @property
    def unique_value_strings(self) -> Iterable[str]:
        return self.info.unique_value_strings


class DataFrameTableBuilderAbstract(TableBuilderAbstract):
    """
    Abstract builder for dataframe info table.

    Parameters
    ----------
    info : DataFrameGlimpseInfo.
        Instance of DataFrameGlimpseInfo.
    """

    def __init__(self, *, info: DataFrameGlimpseInfo) -> None:
        self.info: DataFrameGlimpseInfo = info

    def get_lines(self) -> list[str]:
        self._lines = []
        if self.col_count == 0:
            self._fill_empty_info()
        else:
            self._fill_non_empty_info()
        return self._lines

    def _fill_empty_info(self) -> None:
        """Add lines to the info table, pertaining to empty dataframe."""
        self.add_summary_line()
        self._lines.append(f"Empty {type(self.data).__name__}\n")

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""

    @property
    def data(self) -> DataFrame:
        """DataFrame."""
        return self.info.data

    @property
    def ids(self) -> Index:
        """Dataframe columns."""
        return self.info.ids

    @property
    def col_count(self) -> int:
        """Number of dataframe columns to be summarized."""
        return self.info.col_count

    def add_summary_line(self) -> None:
        """Add line containing type, rows and columns."""
        self._lines.append(f"{type(self.data)} with {len(self.data)} rows and {len(self.data.columns)} columns.")


class TableBuilderMixin(TableBuilderAbstract):
    """
    Mixin for glimpse output.
    """

    SPACING: str = " " * 2
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    include_dtype: bool
    include_non_null_count: bool
    include_null_count: bool
    include_nunique: bool
    unique_values: bool

    @property
    @abstractmethod
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in table."""

    @property
    def header_column_widths(self) -> Sequence[int]:
        """Widths of header columns (only titles)."""
        return [len(col) for col in self.headers]

    def _get_gross_column_widths(self) -> Sequence[int]:
        """Get widths of columns containing both headers and actual content."""
        body_column_widths = self._get_body_column_widths()
        return [
            max(*widths)
            for widths in zip(self.header_column_widths, body_column_widths)
        ]

    def _get_body_column_widths(self) -> Sequence[int]:
        """Get widths of table content columns."""
        strcols: Sequence[Sequence[str]] = list(zip(*self.strrows))
        return [max(len(x) for x in col) for col in strcols]

    @abstractmethod
    def _gen_rows(self) -> Iterator[Sequence[str]]:
        """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
        # TODO: Move _gen_rows() and headers() to here as they are the same!

    def add_header_line(self) -> None:
        header_line = self.SPACING.join(
            [
                _put_str(header, col_width)
                for header, col_width in zip(self.headers, self.gross_column_widths)
            ]
        )
        self._lines.append(header_line)

    def add_separator_line(self) -> None:
        separator_line = self.SPACING.join(
            [
                _put_str("-" * header_colwidth, gross_colwidth)
                for header_colwidth, gross_colwidth in zip(
                    self.header_column_widths, self.gross_column_widths
                )
            ]
        )
        self._lines.append(separator_line)

    def add_body_lines(self) -> None:
        # todo: where should we get 300 from???
        trim_width = min(get_option("display.width"), 300)
        for row in self.strrows:
            body_line = self.SPACING.join(
                [
                    _put_str(col, gross_colwidth)
                    for col, gross_colwidth in zip(row, self.gross_column_widths)
                ]
            )
            body_line = _trim_str(body_line, trim_width)
            self._lines.append(body_line)

    def _gen_dtypes(self) -> Iterator[str]:
        """Iterator with string representation of column dtypes."""
        for dtype in self.dtypes:
            yield pprint_thing(dtype)

    def _gen_non_null_counts(self) -> Iterator[str]:
        """Iterator with string representation of non-null counts."""
        for count in self.non_null_counts:
            yield f"{count} non-null"

    def _gen_null_counts(self) -> Iterator[str]:
        """Iterator with string representation of null counts."""
        for count in self.null_counts:
            yield f"{count} null"

    def _gen_nunique_counts(self) -> Iterator[str]:
        """Iterator with string representation of nunique counts."""
        for n in self.nunique_counts:
            if isinstance(n, int):
                yield f"{n} unique"
            else:
                yield f"{n}"

    def _gen_value_strings(self) -> Iterator[str]:
        """Iterator with string representation of the first values in the columns."""
        for value_string in self.value_strings:
            yield f"{value_string}"

    def _gen_unique_value_strings(self) -> Iterator[str]:
        """Iterator with string representation of the unique values in the columns."""
        for unique_value_string in self.unique_value_strings:
            yield f"{unique_value_string}"


class DataFrameTableBuilder(DataFrameTableBuilderAbstract, TableBuilderMixin):
    """
    Dataframe glimpse table builder.
    """

    def __init__(
        self,
        *,
        info: DataFrameGlimpseInfo,
        include_dtype: bool,
        include_non_null_count: bool,
        include_null_count: bool,
        include_nunique: bool,
        unique_values: bool
    ) -> None:
        self.info = info
        self.include_dtype = include_dtype
        self.include_non_null_count = include_non_null_count
        self.include_null_count = include_null_count
        self.include_nunique = include_nunique
        self.unique_values = unique_values
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_summary_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self._lines.append("")

    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in table."""
        header_list = ["Column"]

        # Dtype
        if self.include_dtype is True:
            header_list.append("Dtype")

        # Non-null count
        if self.include_non_null_count is True:
            header_list.append("Non-null")

        # Null count
        if self.include_null_count is True:
            header_list.append("Null")

        # N-unique
        if self.include_nunique is True:
            header_list.append("N-unique")

        # Unique values
        if self.unique_values is True:
            header_list.append("Unique values")
        else:
            header_list.append("Values")

        return header_list

    def _gen_rows(self) -> Iterator[Sequence[str]]:
        to_include = list()

        # Columns
        to_include.append(self._gen_columns())

        # Dtype
        if self.include_dtype is True:
            to_include.append(self._gen_dtypes())

        # Non-null count
        if self.include_non_null_count is True:
            to_include.append(self._gen_non_null_counts())

        # Null count
        if self.include_null_count is True:
            to_include.append(self._gen_null_counts())

        # N-unique
        if self.include_nunique is True:
            to_include.append(self._gen_nunique_counts())

        # Values
        if self.unique_values is True:
            to_include.append(self._gen_unique_value_strings())
        else:
            to_include.append(self._gen_value_strings())

        yield from zip(
            *to_include
        )

    def _gen_columns(self) -> Iterator[str]:
        """Iterator with string representation of column names."""
        for col in self.ids:
            yield pprint_thing(col)


class SeriesTableBuilderAbstract(TableBuilderAbstract):
    """
    Abstract builder for series info table.

    Parameters
    ----------
    info : SeriesGlimpseInfo.
        Instance of SeriesGlimpseInfo.
    """

    def __init__(self, *, info: SeriesGlimpseInfo) -> None:
        self.info: SeriesGlimpseInfo = info

    def get_lines(self) -> list[str]:
        self._lines = []
        self._fill_non_empty_info()
        return self._lines

    @property
    def data(self) -> Series:
        """Series."""
        return self.info.data

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""


class SeriesTableBuilder(SeriesTableBuilderAbstract, TableBuilderMixin):
    """
    Series info table builder.
    """

    def __init__(
        self,
        *,
        info: SeriesGlimpseInfo,
        include_dtype: bool,
        include_non_null_count: bool,
        include_null_count: bool,
        include_nunique: bool,
        unique_values: bool
    ) -> None:
        self.info = info
        self.include_dtype = include_dtype
        self.include_non_null_count = include_non_null_count
        self.include_null_count = include_null_count
        self.include_nunique = include_nunique
        self.unique_values = unique_values
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty series."""
        self.add_series_name_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()

    def add_series_name_line(self) -> None:
        self._lines.append(f"Series name: {self.data.name}")

    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in table."""
        header_list = ["Name"]

        # Dtype
        if self.include_dtype is True:
            header_list.append("Dtype")

        # Non-null count
        if self.include_non_null_count is True:
            header_list.append("Non-null")

        # Null count
        if self.include_null_count is True:
            header_list.append("Null")

        # N-unique
        if self.include_nunique is True:
            header_list.append("N-unique")

        # Unique values
        if self.unique_values is True:
            header_list.append("Unique values")
        else:
            header_list.append("Values")

        return header_list

    def _gen_rows(self) -> Iterator[Sequence[str]]:
        to_include = list()

        # Name
        to_include.append(self._gen_name())

        # Dtype
        if self.include_dtype is True:
            to_include.append(self._gen_dtypes())

        # Non-null count
        if self.include_non_null_count is True:
            to_include.append(self._gen_non_null_counts())

        # Null count
        if self.include_null_count is True:
            to_include.append(self._gen_null_counts())

        # N-unique
        if self.include_nunique is True:
            to_include.append(self._gen_nunique_counts())

        # Values
        if self.unique_values is True:
            to_include.append(self._gen_unique_value_strings())
        else:
            to_include.append(self._gen_value_strings())

        yield from zip(
            *to_include
        )

    def _gen_name(self) -> Iterator[str]:
        """Iterator with string representation of series name."""
        # todo: should this have a for loop?? (or should I remove the other for loops?)
        yield pprint_thing(self.data.name)



def _get_nunique_without_unhashable_error(df: DataFrame) -> Sequence[int]:
    # imported here to avoid circular imports from partially imported module.
    from pandas import Series

    try:
        s = df.nunique()
    except:
        s = Series(dtype=int)
        for col in df.columns:
            try:
                s[col] = df[col].nunique()
            except:
                s[col] = 'unhashable'

    return s

# todo: why is glimpse not ending with a newline when info is??
# todo: how should glimpse handle an empty dataframe/series?
# todo: should unique_values have a sorting option?
# todo: should _put_str and _trim_str be merged ? (it would make _trim_str less convoluted)
# todo: should I make an abbreviated version?
# todo: series new line after print body


