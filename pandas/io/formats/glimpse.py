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

if TYPE_CHECKING:
    from pandas.core.frame import (
        DataFrame,
        Series,
    )

frame_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',  
    ...                'zeta', 'eta', 'theta', 'iota']
    >>> float_values = [0.0, None, 0.25, None, 0.5, None, 0.75, None,
    ...                 1.0]
    >>> df = pd.DataFrame({"int_col": int_values, "text_col": 
    ...                   text_values, "float_col": float_values})
    >>> df.head(5)
       int_col text_col  float_col
    0        1    alpha       0.00
    1        2     beta        NaN
    2        3    gamma       0.25
    3        1    delta        NaN
    4        2  epsilon       0.50
    
    Prints a glimpse of each column and its dtypes.
    
    >>> df.glimpse()
    DataFrame with 9 rows and 3 columns.
    int_col    <int64>    1, 2, 3, 1, 2, 3, 1, 2, 3                                 
    text_col   <object>   'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',  ...
    float_col  <float64>  0.0, nan, 0.25, nan, 0.5, nan, 0.75, nan, 1.0             
    
    Prints a glimpse of the unique values instead of the first values.

    >>> df.glimpse(unique_values=True)
    DataFrame with 9 rows and 3 columns.
    int_col    <int64>    1, 2, 3                                                   
    text_col   <object>   'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',  ...
    float_col  <float64>  0.0, nan, 0.25, 0.5, 0.75, 1.0                            
    
    Adds the null and non-null counts to the glimpse. This will change  
    to the verbose output format.
    
    >>> df.glimpse(isna=True, notna=True)
    DataFrame with 9 rows and 3 columns.
    Column     Dtype    Null    Non-null    Values                                                                   
    ------     -----    ----    --------    ------                                                                   
    int_col    int64    0 null  9 non-null  1, 2, 3, 1, 2, 3, 1, 2, 3               
    text_col   object   0 null  9 non-null  'alpha', 'beta', 'gamma', 'delta', ' ...
    float_col  float64  4 null  5 non-null  0.0, nan, 0.25, nan, 0.5, nan, 0.75, ...
    
    Adds the column index and the number of unique values to the
    glimpse.
    
    >>> df.glimpse(index=True, nunique=True)
    DataFrame with 9 rows and 3 columns.
     #   Column     Dtype    N-unique  Values                                                                   
    ---  ------     -----    --------  ------                                                                   
     0   int_col    int64    3 unique  1, 2, 3, 1, 2, 3, 1, 2, 3                    
     1   text_col   object   9 unique  'alpha', 'beta', 'gamma', 'delta', 'epsil ...
     2   float_col  float64  5 unique  0.0, nan, 0.25, nan, 0.5, nan, 0.75, nan, ...
    
    Adds the null, non-null, and nunique counts while retaining the 
    non-verbose format.
    
    >>> df.glimpse(isna=True, notna=True, nunique=True, verbose=False)
    DataFrame with 9 rows and 3 columns.
    int_col    <int64>    (0/9)  |3|  1, 2, 3, 1, 2, 3, 1, 2, 3                     
    text_col   <object>   (0/9)  |9|  'alpha', 'beta', 'gamma', 'delta', 'epsilo ...
    float_col  <float64>  (4/5)  |5|  0.0, nan, 0.25, nan, 0.5, nan, 0.75, nan,  ..."""
)

frame_unique_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon',
    ...                'zeta', 'eta', 'theta', 'iota']
    >>> float_values = [0.0, None, 0.25, None, 0.5, None, 0.75, None, 
    ...                 1.0]
    >>> df = pd.DataFrame({"int_col": int_values, "text_col": 
    ...                   text_values, "float_col": float_values})
    >>> df.head(5)
       int_col text_col  float_col
    0        1    alpha       0.00
    1        2     beta        NaN
    2        3    gamma       0.25
    3        1    delta        NaN
    4        2  epsilon       0.50

    Prints a glimpse of each column with its dtype and unique values.

    >>> df.glimpse_unique()
    DataFrame with 9 rows and 3 columns.
    int_col    <int64>    1, 2, 3                                                   
    text_col   <object>   'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',  ...
    float_col  <float64>  0.0, nan, 0.25, 0.5, 0.75, 1.0    

    Adds the null and non-null counts to the glimpse. This will change 
    to the verbose output format.

    >>> df.glimpse_unique(isna=True, notna=True)
    DataFrame with 9 rows and 3 columns.
    Column     Dtype    Null    Non-null    Unique values                                                            
    ------     -----    ----    --------    -------------                                                            
    int_col    int64    0 null  9 non-null  1, 2, 3                                 
    text_col   object   0 null  9 non-null  'alpha', 'beta', 'gamma', 'delta', ' ...
    float_col  float64  4 null  5 non-null  0.0, nan, 0.25, 0.5, 0.75, 1.0"""
)

series_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'alpha', 'beta']
    >>> s = pd.Series(text_values, index=int_values, 
                      name='greek_letters')

    Prints a glimpse of the Series with its dtype.

    >>> s.glimpse()
    Series (greek_letters) with 5 rows.
    greek_letters  <object>  'alpha', 'beta', 'gamma', 'alpha', 'beta'

    Prints a glimpse of the unique values instead of the first values.

    s.glimpse(unique_values=True)
    Series (greek_letters) with 6 rows.
    greek_letters  <object>  'alpha', 'beta', 'gamma'"""
)

series_unique_examples_sub = dedent(
    """\
    >>> int_values = [1, 2, 3, 4, 5]
    >>> text_values = ['alpha', 'beta', 'gamma', 'alpha', 'beta']
    >>> s = pd.Series(text_values, index=int_values, 
                      name='greek_letters')

    Prints a glimpse of the Series with its dtype and unique values.

    >>> s.glimpse_unique()
    Series (greek_letters) with 5 rows.
    greek_letters  <object>  'alpha', 'beta', 'gamma'"""
)

frame_see_also_sub = dedent(
    """\
    DataFrame.glimpse_unique: Print a transposed glimpse of a DataFrame 
        with its unique values.
    DataFrame.info: Print a concise summary of a DataFrame.
    DataFrame.describe: Generate descriptive statistics of DataFrame
        columns."""
)

frame_unique_see_also_sub = dedent(
    """\
    DataFrame.glimpse: Print a transposed glimpse of a DataFrame with 
        its underlying data.
    DataFrame.info: Print a concise summary of a DataFrame.
    DataFrame.describe: Generate descriptive statistics of DataFrame
        columns."""
)

series_see_also_sub = dedent(
    """\
    Series.glimpse_unique: Print a transposed glimpse of a Series
        with its unique values.
    Series.info: Print a concise summary of a Series.
    Series.describe: Generate descriptive statistics of Series."""
)

series_unique_see_also_sub = dedent(
    """\
    Series.glimpse: Print a transposed glimpse of a Series with its 
        underlying data.
    Series.info: Print a concise summary of a Series.
    Series.describe: Generate descriptive statistics of Series."""
)

index_sub = dedent(
    """\
    index : bool, optional
        Whether to print the column indices.\n"""
)

unique_values_sub = dedent(
    """\
    unique_values: bool, optional
        Whether to print a glimpse of the unique values instead of the first values.\n"""
)

frame_sub_kwargs_glimpse = {
    "klass": "DataFrame",
    "columns_sub": "s of the columns",
    "unique_sub": "underlying data",
    "index_sub": index_sub,
    "unique_values_sub": unique_values_sub,
    "examples_sub": frame_examples_sub,
    "see_also_sub": frame_see_also_sub
}

series_sub_kwargs_glimpse = {
    "klass": "Series",
    "columns_sub": "",
    "unique_sub": "underlying data",
    "index_sub": "",
    "unique_values_sub": unique_values_sub,
    "examples_sub": series_examples_sub,
    "see_also_sub": series_see_also_sub
}

frame_sub_kwargs_glimpse_unique = {
    "klass": "DataFrame",
    "columns_sub": "s of the columns",
    "unique_sub": "unique values",
    "index_sub": index_sub,
    "unique_values_sub": "",
    "examples_sub": frame_unique_examples_sub,
    "see_also_sub": frame_unique_see_also_sub
}

series_sub_kwargs_glimpse_unique = {
    "klass": "Series",
    "columns_sub": "",
    "unique_sub": "unique values",
    "index_sub": "",
    "unique_values_sub": "",
    "examples_sub": series_unique_examples_sub,
    "see_also_sub": series_unique_see_also_sub
}

GLIMPSE_DOCSTRING = dedent(
    """
    Print a transposed glimpse of a {klass} with its {unique_sub}.
    
    This method prints a transposed version of a {klass} with
    columns running down the page, and a preview of the data running 
    across. Further, it can include extra information such as the index,
    dtype, null values, non-null values, and number of unique values.
    
    ..versionadded:: 1.5.3
    
    Parameters
    ----------
    {index_sub}dtype : bool, optional
        Whether to print the dtype{columns_sub}.
    isna : bool, optional
        Whether to print the null count{columns_sub}.
    notna : bool, optional
        Whether to print the non-null count{columns_sub}.
    nunique: bool, optional
        Whether to print the number of unique values.
    {unique_values_sub}verbose : bool, optional
        Whether to print the headers and count descriptions. By default,
        the setting goes to false if only dtype is enabled otherwise it
        goes to true.
    emphasize: bool, optional
        Whether to emphasize the optional information columns. By 
        default, it is enabled if verbose is false.
    buf : writable buffer, defaults to sys.stdout
        Where to send the output. By default, the output is printed to
        sys.stdout. Pass a writable buffer if you need to further
        process the output.
    width : int, optional
        The width at which the output is trimmed. By default, the width
        is determined by the pandas display.width option.\
        
    Returns
    -------
    None
        This method prints a glimpse of a {klass} and returns None.

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
    >>> pd.io.formats.glimpse._put_str("panda", 6)
    'panda '
    >>> pd.io.formats.glimpse._put_str("panda", 4)
    'pand'
    """
    return str(s)[:space].ljust(space)


def _trim_str(s: str, length: int) -> str:
    """
    Crop a string from the right and append ' ...' if it is too long.

    Parameters
    ----------
    s : str
        String to be formatted.
    length : int
        Length to force string to be of.

    Returns
    -------
    str
        String of length no longer than desired length.

    Examples
    --------
    >>> pd.io.formats.glimpse._trim_str("red pandas", 10)
    'red pandas'
    >>> pd.io.formats.glimpse._trim_str("red pandas", 8)
    'red pa ...'
    >>> pd.io.formats.glimpse._trim_str("pandas    ", 8)
    'pandas    '
    """
    if len(str(s)) > length:
        if s[length-4:length] == "    ":
            return str(s[:length])
        else:
            return f"{s[:length - 4]} ..."
    else:
        return str(s)



def _format_body_line(s: str,
                      col_widths: Sequence[int],
                      spacing_width: int,
                      start_col_idx: int,
                      end_col_idx: int,
                      bold: bool = False,
                      italic: bool = False):
    """
    Add escape characters to one or more columns in the glimpse output
    to make it appear bold and/or italic.

    Parameters
    ----------
    s : str
        String to be formatted.
    col_widths : Sequence[int]
        The widths of the columns.
    spacing_width : int
        The width of the spacing.
    start_col_idx : int
        The first column to be emphasized.
    end_col_idx : int
        The final column to be emphasized.
    bold : bool
        Whether to make the string bold.
    italic : bool
        Whether to make the string italic.

    Returns
    -------
    str
        The provided string with desired escape characters.
    """

    start_pos = sum(col_widths[:start_col_idx]) \
                + spacing_width * (start_col_idx)
    end_pos = sum(col_widths[:end_col_idx]) \
              + spacing_width * (end_col_idx - 1)

    if (bold and italic) is True:
        return s[:start_pos] + "\033[1;3m" + s[start_pos:end_pos] + "\033[0m" + s[end_pos:]
    elif bold is True:
        return s[:start_pos] + "\033[1m" + s[start_pos:end_pos] + "\033[0m" + s[end_pos:]
    elif italic is True:
        return s[:start_pos] + "\033[3m" + s[start_pos:end_pos] + "\033[0m" + s[end_pos:]
    else:
        return str(s)

def _auto_format_body_line(s: str,
                           col_widths: Sequence[int],
                           spacing_width: int,
                           bold: bool = False,
                           italic: bool = False,
                           include_index=None,
                           include_dtype=None,
                           include_null_count=None,
                           include_non_null_count=None,
                           include_nunique=None,
                           is_verbose=None):
    """
    Wrapper for _format_body_line, which automatically infers
    start_col_idx and end_col_idx.

    Parameters
    ----------
    s : str
        String to be formatted.
    col_widths : Sequence[int]
        The widths of the columns.
    spacing_width : int
        The width of the spacing.
    bold : bool
        Whether to make the string bold.
    italic : bool
        Whether to make the string italic.
    include_index : bool
        Whether index is included in the glimpse.
    include_dtype : bool
        Whether dtype is included in the glimpse.
    include_null_count : bool
        Whether null count is included in the glimpse.
    include_non_null_count : bool
        Whether non null count is included in the glimpse.
    include_nunique : bool
        Whether nunique is included in the glimpse.
    is_verbose : bool
        Whether the glimpse is in its verbose form.

    Returns
    -------
    str
        The provided string with desired escape characters.
    """

    start_col_idx = 1 + include_index

    end_col_idx = start_col_idx + include_dtype + include_nunique +\
                  (include_null_count or include_non_null_count)

    # Edge-case with verbose=True, emphasize=True, isna=Ture, notna=True
    if is_verbose is True and include_null_count is True and include_non_null_count is True:
        end_col_idx += 1

    if end_col_idx > start_col_idx:
        return _format_body_line(s, col_widths, spacing_width,
                                 start_col_idx=start_col_idx,
                                 end_col_idx=end_col_idx,
                                 italic=italic,
                                 bold=bold)
    else:
        return str(s)


class BaseGlimpseInfo(ABC):
    """
    Base class for DataFrameGlimpseInfo and SeriesGlimpseInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either Dataframe or Series.
    glimpse_width: int
        The glimpse print width.
    """

    data: DataFrame | Series
    glimpse_width: int

    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's column(s).
        """

    @property
    @abstractmethod
    def null_counts(self) -> Sequence[int]:
        """Sequence of null counts for all column(s)."""

    @property
    @abstractmethod
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all column(s)."""

    @property
    @abstractmethod
    def nunique_counts(self) -> Sequence[int | str]:
        """Sequence of counts of unique element for all column(s)."""

    @property
    @abstractmethod
    def value_strings(self) -> Sequence[str]:
        """Sequence of strings representing the values."""

    @property
    @abstractmethod
    def unique_value_strings(self) -> Sequence[str]:
        """Sequence of strings representing the unique values."""

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        """Boolean representing if the data is empty."""

    @abstractmethod
    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        dtype: bool | None,
        isna: bool | None,
        notna: bool | None,
        nunique: bool | None,
        unique_values: bool | None,
        verbose: bool | None,
        emphasize: bool | None,
    ) -> None:
        pass


class DataFrameGlimpseInfo(BaseGlimpseInfo):
    """
    Class storing DataFrame-specific info needed to glimpse.
    """

    def __init__(
        self,
        data: DataFrame,
        glimpse_width: int | None,
    ) -> None:
        self.data: DataFrame = data
        self.glimpse_width = get_option("display.width") if \
            glimpse_width is None else glimpse_width

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
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all column(s)."""
        return self.data.count()

    @property
    def null_counts(self) -> Sequence[int]:
        """Sequence of null counts for all column(s)."""
        return self.data.isna().sum()

    @property
    def nunique_counts(self) -> Sequence[int | str]:
        """Sequence of counts of unique elements for all column(s)."""
        try:
            s = self.data.nunique()
        except TypeError:
            from pandas import Series

            s = Series(dtype=int)
            for col in self.data.columns:
                try:
                    s[col] = self.data[col].nunique()
                except TypeError:
                    s[col] = 'unhashable'
        return s

    @property
    def value_strings(self) -> Sequence[str]:
        """Sequence of strings representing the values."""
        from pandas import Series

        # calculate max number of elements to include in the glimpse.
        # this greatly reduces runtime.
        number_of_elements_to_include = int(1 + self.glimpse_width / 3)

        s = Series(dtype=str)
        for i, col in enumerate(self.ids):
            s[f'{col}_{i}'] = ', '.join(map(
                lambda x: pprint_thing(x, quote_strings=True),
                self.data.iloc[:, i].head(number_of_elements_to_include).to_list()
            ))[:self.glimpse_width]
        return s

    @property
    def unique_value_strings(self) -> Sequence[str]:
        """Sequence of strings representing the unique values."""
        from pandas import Series

        s = Series(dtype=str)
        for i, col in enumerate(self.ids):
            try:
                s[f'{col}_{i}'] = ', '.join(map(
                    lambda x: pprint_thing(x, quote_strings=True),
                    self.data.iloc[:, i].unique()
                ))[:self.glimpse_width]
            except TypeError:
                s[f'{col}_{i}'] = 'unhashable'
        return s

    @property
    def is_empty(self) -> bool:
        return self.data.empty

    def render(
        self,
        *,
        buf: WriteBuffer[str] | None,
        index: bool | None,
        dtype: bool | None,
        isna: bool | None,
        notna: bool | None,
        nunique: bool | None,
        unique_values: bool | None,
        verbose: bool | None,
        emphasize: bool | None,
    ) -> None:
        printer = DataFrameGlimpsePrinter(
            info=self,
            include_index=index,
            include_dtype=dtype,
            include_null_count=isna,
            include_non_null_count=notna,
            include_nunique=nunique,
            unique_values=unique_values,
            verbose=verbose,
            emphasize=emphasize,
        )
        printer.to_buffer(buf)


class SeriesGlimpseInfo(BaseGlimpseInfo):
    """
    Class storing Series-specific info needed to glimpse.
    """

    def __init__(
        self,
        data: Series,
        glimpse_width: int | None
    ) -> None:
        self.data: Series = data
        self.glimpse_width = get_option("display.width") if \
            glimpse_width is None else glimpse_width

    def render(
            self,
            *,
            buf: WriteBuffer[str] | None,
            dtype: bool | None,
            isna: bool | None,
            notna: bool | None,
            nunique: bool | None,
            unique_values: bool | None,
            verbose: bool | None,
            emphasize: bool | None,
    ) -> None:
        printer = SeriesGlimpsePrinter(
            info=self,
            include_dtype=dtype,
            include_null_count=isna,
            include_non_null_count=notna,
            include_nunique=nunique,
            unique_values=unique_values,
            verbose=verbose,
            emphasize=emphasize,
        )
        printer.to_buffer(buf)

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """Sequence of dtypes for all column(s)."""
        return [self.data.dtypes]

    @property
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all column(s)."""
        return [self.data.count()]

    @property
    def null_counts(self) -> Sequence[int]:
        """Sequence of null counts for all column(s)."""
        return [self.data.isna().sum()]

    @property
    def nunique_counts(self) -> Sequence[int | str]:
        """Sequence of counts of unique elements for all column(s)."""
        return [self.data.nunique()]

    @property
    def value_strings(self) -> Sequence[str]:
        """Sequence of strings representing the values."""
        from pandas import Series

        # calculate max number of elements to include in the glimpse.
        # this greatly reduces runtime.
        number_of_elements_to_include = int(1 + self.glimpse_width / 3)

        s = Series(dtype=str)
        s[self.data.name] = ', '.join(map(
                lambda x: pprint_thing(x, quote_strings=True),
                self.data.head(number_of_elements_to_include).to_list()
            ))[:self.glimpse_width]
        return s

    @property
    def unique_value_strings(self) -> Sequence[str]:
        """Sequence of strings representing the unique values."""
        from pandas import Series

        s = Series(dtype=str)
        s[self.data.name] = ', '.join(map(
                lambda x: pprint_thing(x, quote_strings=True),
                self.data.unique()
            ))[:self.glimpse_width]
        return s

    @property
    def is_empty(self) -> bool:
        return self.data.empty


class GlimpsePrinterAbstract:
    """
    Class for printing DataFrame or Series glimpse.
    """

    def to_buffer(self, buf: WriteBuffer[str] | None = None) -> None:
        """Save dataframe glimpse into buffer."""
        table_builder = self._create_table_builder()
        lines = table_builder.get_lines()
        if buf is None:
            buf = sys.stdout
        fmt.buffer_put_lines(buf, lines)

    @abstractmethod
    def _create_table_builder(self) -> TableBuilderAbstract:
        """Create instance of table builder."""


class DataFrameGlimpsePrinter(GlimpsePrinterAbstract):
    """
    Class for printing DataFrame glimpse.

    This class outputs the glimpse to desired buffer and interprets
    configuration inputs from the user.

    Parameters
    ----------
    info : DataFrameGlimpseInfo
        Instance of DataFrameGlimpseInfo.
    include_index: bool, optional
        Whether to show the index.
    include_dtype: bool, optional
        Whether to show the dtypes.
    include_null_count: bool, optional
        Whether to show the null counts.
    include_non_null_count: bool, optional
        Whether to show the non-null counts.
    include_nunique: bool, optional
        Whether to show the number of unique values.
    unique_values: bool, optional
        Whether to show the unique values.
    verbose: bool, optional
        Whether to print the headers and units. By default, the
        setting goes to false if only dtype is enabled otherwise
        it goes to true.
    emphasize: bool, optional
        Whether to emphasize the columns between the column name
        and the values. By default, it is enabled if verbose is
        false.
    """

    def __init__(
        self,
        info: DataFrameGlimpseInfo,
        include_index: bool | None = None,
        include_dtype: bool | None = None,
        include_null_count: bool | None = None,
        include_non_null_count: bool | None = None,
        include_nunique: bool | None = None,
        unique_values: bool | None = None,
        verbose: bool | None = None,
        emphasize: bool | None = None,
    ) -> None:
        self.info = info
        self.data = info.data
        self.include_index = self._initialize_index(include_index)
        self.include_dtype = self._initialize_dtype(include_dtype)
        self.include_non_null_count = self._initialize_non_null_count(include_non_null_count)
        self.include_null_count = self._initialize_null_count(include_null_count)
        self.include_nunique = self._initialize_nunique(include_nunique)
        self.unique_values = self._initialize_unique_values(unique_values)
        self.verbose = self._initialize_verbose(verbose)
        self.emphasize = self._initialize_emphasize(emphasize)

    @property
    def is_empty(self) -> bool:
        return self.info.is_empty

    def _initialize_verbose(self, verbose: bool | None) -> bool:
        if verbose is not None:
            return verbose

        # infer the default value of verbose if it was not provided.
        if (self.include_index or
            self.include_non_null_count or
            self.include_null_count or
            self.include_nunique) is False:
            return False
        else:
            return True

    def _initialize_emphasize(self, emphasize: bool | None) -> bool:
        if emphasize is None:
            return not self.verbose
        else:
            return emphasize

    def _initialize_index(self, include_index: bool | None) -> bool:
        if include_index is None:
            return False
        else:
            return include_index

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
        Create instance of table builder based on desired columns in
        the glimpse.
        """
        return DataFrameTableBuilder(
            info=self.info,
            include_index=self.include_index,
            include_dtype=self.include_dtype,
            include_null_count=self.include_null_count,
            include_non_null_count=self.include_non_null_count,
            include_nunique=self.include_nunique,
            unique_values=self.unique_values,
            verbose=self.verbose,
            emphasize=self.emphasize,
        )


class SeriesGlimpsePrinter(GlimpsePrinterAbstract):
    """Class for printing series info.

    This class outputs the glimpse to desired buffer and interprets
    configuration inputs from the user.

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
    verbose: bool, optional
        Whether to print the headers and units. By default, the
        setting goes to false if only dtype is enabled otherwise
        it goes to true.
    emphasize: bool, optional
        Whether to emphasize the columns between the column name
        and the values. By default, it is enabled if verbose is
        false.
    """

    def __init__(
        self,
        info: SeriesGlimpseInfo,
        include_dtype: bool | None = None,
        include_null_count: bool | None = None,
        include_non_null_count: bool | None = None,
        include_nunique: bool | None = None,
        unique_values: bool | None = None,
        verbose: bool | None = None,
        emphasize: bool | None = None,
    ) -> None:
        self.info = info
        self.data = info.data
        self.include_dtype = self._initialize_dtype(include_dtype)
        self.include_null_count = self._initialize_null_count(include_null_count)
        self.include_non_null_count = self._initialize_non_null_count(include_non_null_count)
        self.include_nunique = self._initialize_nunique(include_nunique)
        self.unique_values = self._initialize_unique_values(unique_values)
        self.verbose = self._initialize_verbose(verbose)
        self.emphasize = self._initialize_emphasize(emphasize)

    @property
    def is_empty(self) -> bool:
        return self.info.is_empty

    def _initialize_verbose(self, verbose: bool | None) -> bool:
        if verbose is not None:
            return verbose

        # infer the default value of verbose if it was not provided.
        if (self.include_non_null_count or
            self.include_null_count or
            self.include_nunique) is False:
            return False
        else:
            return True

    def _initialize_emphasize(self, emphasize: bool | None) -> bool:
        if emphasize is None:
            return not self.verbose
        else:
            return emphasize

    def _initialize_index(self, include_index: bool | None) -> bool:
        if include_index is None:
            return False
        else:
            return include_index

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
        Create instance of table builder based on desired columns
        in the glimpse.
        """
        return SeriesTableBuilder(
            info=self.info,
            include_dtype=self.include_dtype,
            include_non_null_count=self.include_non_null_count,
            include_null_count=self.include_null_count,
            include_nunique=self.include_nunique,
            unique_values=self.unique_values,
            verbose=self.verbose,
            emphasize=self.emphasize,
        )


class TableBuilderAbstract(ABC):
    """
    Abstract builder for glimpse table.

    This class contains the coarse building blocks needed to generate
    the glimpse.

    Parameters
    ----------
    info : BaseGlimpseInfo
        Instance of either DataFrameGlimpseInfo or SeriesGlimpseInfo.
    """

    _lines: list[str]
    info: BaseGlimpseInfo

    def get_lines(self) -> list[str]:
        """Construct each line to print in the glimpse."""
        self._lines = []
        if self.is_empty is True:
            self._fill_empty_glimpse()
        else:
            self._fill_non_empty_glimpse()
        return self._lines

    @property
    def data(self) -> DataFrame | Series:
        return self.info.data

    @property
    def dtypes(self) -> Iterable[Dtype]:
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

    @property
    def is_empty(self) -> bool:
        """Boolean representing if the data is empty."""
        return self.info.is_empty

    @abstractmethod
    def add_summary_line(self) -> None:
        """Add line containing type, rows and columns."""

    def _fill_empty_glimpse(self) -> None:
        """
        Add lines to the glimpse table, pertaining to empty DataFrames.
        """
        self.add_summary_line()
        self._lines.append("")

    @abstractmethod
    def _fill_non_empty_glimpse(self) -> None:
        """
        Add lines to the glimpse table, pertaining to non-empty
        DataFrames or Series.
        """


class DataFrameTableBuilderAbstract(TableBuilderAbstract):
    """
    Abstract builder for dataframe glimpse table.

    This class contains the coarse building blocks needed to generate
    the glimpse.

    Parameters
    ----------
    info : DataFrameGlimpseInfo.
        Instance of DataFrameGlimpseInfo.
    """

    def __init__(self, *, info: DataFrameGlimpseInfo) -> None:
        self.info: DataFrameGlimpseInfo = info

    @property
    def data(self) -> DataFrame:
        return self.info.data

    @property
    def ids(self) -> Index:
        """The Dataframe columns."""
        return self.info.ids

    def add_summary_line(self) -> None:
        """Add line containing type, and number of rows and columns."""
        self._lines.append(f"{type(self.data).__name__} with "
                           f"{len(self.data)} rows and "
                           f"{len(self.data.columns)} columns.")


class TableBuilderMixin(TableBuilderAbstract):
    """
    Mixin for glimpse output.

    This class generates each element in the glimpse by using
    information from DataFrame- or SeriesTableBuilderAbstract.
    """

    SPACING: str = " " * 2
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    trim_width: int
    include_index: bool
    include_dtype: bool
    include_null_count: bool
    include_non_null_count: bool
    include_nunique: bool
    unique_values: bool
    verbose: bool
    emphasize: bool

    @property
    @abstractmethod
    def headers(self) -> Sequence[str]:
        """Header names of the columns in table."""

    @property
    def header_column_widths(self) -> Sequence[int]:
        """Widths of header columns (only the titles)."""
        return [len(col) for col in self.headers]

    def _get_gross_column_widths(self) -> Sequence[int]:
        """
        Get widths of columns containing both headers and actual
        content.
        """
        body_column_widths = self._get_body_column_widths()

        if self.verbose is False:
            return body_column_widths
        else:
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
        Generator function yielding the content of each row.

        Each element represents a row comprising a sequence of strings.
        """

    @abstractmethod
    def _emphasize_body_line(self, body_line) -> str:
        """
        Emphasize some columns based on verbosity and the included
        columns.
        """

    def add_header_line(self) -> None:
        header_line = self.SPACING.join(
            [
                _put_str(header, col_width)
                for header, col_width in zip(self.headers, self.gross_column_widths)
            ]
        )
        header_line = _trim_str(header_line, self.trim_width)
        self._lines.append(header_line)

    def add_separator_line(self) -> None:
        separator_line = self.SPACING.join(
            [
                _put_str("-" * header_colwidth, gross_colwidth)
                for header_colwidth, gross_colwidth in zip(
                self.header_column_widths, self.gross_column_widths)
            ]
        )
        separator_line = _trim_str(separator_line, self.trim_width)
        self._lines.append(separator_line)

    def add_body_lines(self) -> None:
        for row in self.strrows:
            body_line = self.SPACING.join(
                [
                    _put_str(col, gross_colwidth)
                    for col, gross_colwidth in zip(row, self.gross_column_widths)
                ]
            )
            body_line = _trim_str(body_line, self.trim_width)

            # italicize the extra columns.
            # Note: this needs to happen after the gross width have been
            #  used to ensure consistent "visual" width.
            if self.emphasize is True:
                body_line = self._emphasize_body_line(body_line)

            self._lines.append(body_line)

    def _gen_dtypes(self, verbose: bool) -> Iterator[str]:
        """Iterator with string representation of column dtypes."""
        if verbose is True:
            for dtype in self.dtypes:
                yield pprint_thing(dtype)
        else:
            for dtype in self.dtypes:
                yield f"<{pprint_thing(dtype)}>"

    def _gen_null_counts(self) -> Iterator[str]:
        """Iterator with verbose string representation of null counts."""
        for count in self.null_counts:
            yield f"{count} null"

    def _gen_non_null_counts(self) -> Iterator[str]:
        """Iterator with verbose string representation of non-null counts."""
        for count in self.non_null_counts:
            yield f"{count} non-null"

    def _gen_null_and_non_null_counts(self):
        """Iterator with non-verbose string representation of null and non-null counts."""
        for null, non_null in zip(self.null_counts, self.non_null_counts):
            yield f"({null}/{non_null})"

    def _gen_nunique_counts(self, verbose: bool) -> Iterator[str]:
        """Iterator with string representation of nunique counts."""
        if verbose is True:
            for n in self.nunique_counts:
                if isinstance(n, int):
                    yield f"{n} unique"
                else:
                    yield f"{n}"
        else:
            for n in self.nunique_counts:
                yield f"|{n}|"

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
    DataFrame glimpse table builder.
    """

    def __init__(
        self,
        *,
        info: DataFrameGlimpseInfo,
        include_index: bool,
        include_dtype: bool,
        include_null_count: bool,
        include_non_null_count: bool,
        include_nunique: bool,
        unique_values: bool,
        verbose: bool,
        emphasize: bool,
    ) -> None:
        self.info = info
        self.include_index = include_index
        self.include_dtype = include_dtype
        self.include_null_count = include_null_count
        self.include_non_null_count = include_non_null_count
        self.include_nunique = include_nunique
        self.unique_values = unique_values
        self.verbose = verbose
        self.emphasize = emphasize
        self.trim_width = self.info.glimpse_width
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_glimpse(self) -> None:
        """
        Add lines to the glimpse table.
        """
        self.add_summary_line()
        if self.verbose is True:
            self.add_header_line()
            self.add_separator_line()
        self.add_body_lines()
        self._lines.append("")

    @property
    def headers(self) -> Sequence[str]:
        """Header names of the columns in table."""
        header_list = list()

        if self.include_index is True:
            header_list.append(" # ")

        header_list.append("Column")

        if self.include_dtype is True:
            header_list.append("Dtype")

        if self.include_null_count is True:
            header_list.append("Null")

        if self.include_non_null_count is True:
            header_list.append("Non-null")

        if self.include_nunique is True:
            header_list.append("N-unique")

        if self.unique_values is True:
            header_list.append("Unique values")
        else:
            header_list.append("Values")

        return header_list

    def _gen_rows(self) -> Iterator[Sequence[str]]:
        to_include = list()

        if self.include_index is True:
            to_include.append(self._gen_indexes(self.verbose))

        to_include.append(self._gen_columns())

        if self.include_dtype is True:
            to_include.append(self._gen_dtypes(self.verbose))

        # Non-null and null count.
        # Note: If verbose is False, then non-null and null counts can only be included together.
        if self.verbose is True:
            if self.include_null_count is True:
                to_include.append(self._gen_null_counts())

            if self.include_non_null_count is True:
                to_include.append(self._gen_non_null_counts())
        else:
            if (self.include_non_null_count or self.include_null_count) is True:
                to_include.append(self._gen_null_and_non_null_counts())

        if self.include_nunique is True:
            to_include.append(self._gen_nunique_counts(self.verbose))

        if self.unique_values is True:
            to_include.append(self._gen_unique_value_strings())
        else:
            to_include.append(self._gen_value_strings())

        yield from zip(
            *to_include
        )

    def _emphasize_body_line(self, body_line) -> str:
        """Emphasize some columns based on verbosity and the columns."""

        return _auto_format_body_line(body_line,
                                      self.gross_column_widths,
                                      len(self.SPACING),
                                      italic=True,
                                      bold=False,
                                      include_index=self.include_index,
                                      include_dtype=self.include_dtype,
                                      include_null_count=self.include_null_count,
                                      include_non_null_count=self.include_non_null_count,
                                      include_nunique=self.include_nunique,
                                      is_verbose=self.verbose)

    def _gen_indexes(self, verbose: bool) -> Iterator[str]:
        """Iterator with string representation of column index."""
        if verbose is True:
            for i, _ in enumerate(self.ids):
                yield f" {i}"
        else:
            for i, _ in enumerate(self.ids):
                yield f"{i}"

    def _gen_columns(self) -> Iterator[str]:
        """Iterator with string representation of column names."""
        for col in self.ids:
            yield pprint_thing(col)



class SeriesTableBuilderAbstract(TableBuilderAbstract):
    """
    Abstract builder for Series glimpse table.

    This class contains the coarse building blocks needed to generate
    the glimpse.

    Parameters
    ----------
    info : SeriesGlimpseInfo.
        Instance of SeriesGlimpseInfo.
    """

    def __init__(self, *, info: SeriesGlimpseInfo) -> None:
        self.info: SeriesGlimpseInfo = info

    @property
    def data(self) -> Series:
        return self.info.data

    def add_summary_line(self) -> None:
        """Add line containing type, rows and columns."""
        self._lines.append(
            f"{type(self.data).__name__} ({self.data.name}) with {len(self.data)} rows.")


class SeriesTableBuilder(SeriesTableBuilderAbstract, TableBuilderMixin):
    """
    Series glimpse table builder.
    """
    def __init__(
        self,
        *,
        info: SeriesGlimpseInfo,
        include_dtype: bool,
        include_null_count: bool,
        include_non_null_count: bool,
        include_nunique: bool,
        unique_values: bool,
        verbose: bool,
        emphasize: bool,
    ) -> None:
        self.info = info
        self.include_dtype = include_dtype
        self.include_null_count = include_null_count
        self.include_non_null_count = include_non_null_count
        self.include_nunique = include_nunique
        self.unique_values = unique_values
        self.verbose = verbose
        self.emphasize = emphasize
        self.trim_width = self.info.glimpse_width
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_glimpse(self) -> None:
        """
        Add lines to the glimpse table.
        """
        self.add_summary_line()
        if self.verbose is True:
            self.add_header_line()
            self.add_separator_line()
        self.add_body_lines()
        self._lines.append("")


    @property
    def headers(self) -> Sequence[str]:
        """Header names of the columns in table."""
        header_list = ["Name"]

        if self.include_dtype is True:
            header_list.append("Dtype")

        if self.include_non_null_count is True:
            header_list.append("Non-null")

        if self.include_null_count is True:
            header_list.append("Null")

        if self.include_nunique is True:
            header_list.append("N-unique")

        if self.unique_values is True:
            header_list.append("Unique values")
        else:
            header_list.append("Values")

        return header_list

    def _gen_rows(self) -> Iterator[Sequence[str]]:
        to_include = list()

        to_include.append(self._gen_name())

        if self.include_dtype is True:
            to_include.append(self._gen_dtypes(self.verbose))

        # Non-null and null count.
        # Note: If verbose is False, then non-null and null counts can
        #   only be included together.
        if self.verbose is True:
            if self.include_non_null_count is True:
                to_include.append(self._gen_non_null_counts())

            if self.include_null_count is True:
                to_include.append(self._gen_null_counts())
        else:
            if (self.include_non_null_count or self.include_null_count) is True:
                to_include.append(self._gen_null_and_non_null_counts())

        if self.include_nunique is True:
            to_include.append(self._gen_nunique_counts(self.verbose))

        if self.unique_values is True:
            to_include.append(self._gen_unique_value_strings())
        else:
            to_include.append(self._gen_value_strings())

        yield from zip(
            *to_include
        )

    def _emphasize_body_line(self, body_line) -> str:
        """Emphasize some columns based on verbosity and the columns."""
        return _auto_format_body_line(body_line,
                                      self.gross_column_widths,
                                      len(self.SPACING),
                                      italic=True,
                                      bold=False,
                                      include_index=False,
                                      include_dtype=self.include_dtype,
                                      include_null_count=self.include_null_count,
                                      include_non_null_count=self.include_non_null_count,
                                      include_nunique=self.include_nunique,
                                      is_verbose=self.verbose)
        return body_line

    def _gen_name(self) -> Iterator[str]:
        """Iterator with string representation of series name."""
        yield pprint_thing(self.data.name)

