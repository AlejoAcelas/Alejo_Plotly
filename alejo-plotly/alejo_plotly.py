# %%
import torch
import numpy as np
import pandas as pd
import einops

import plotly.express as px
from plotly.graph_objects import Figure

from torch import Tensor
from typing import List, Tuple, Dict, Callable, Union, Optional, Any, overload

from transformer_lens.utils import to_numpy as transformer_lens_to_numpy
from itertools import product
import inspect
from dataclasses import dataclass, field

"""Plolty utils that receive tensors and broadcast them to a long-shape dataframe.
By assigning labels corresponding to data dimensions it makes it easy to integrate
plotly options like color, facet_col and animation_frame to multidimensional data."""

DEFAULT_DIM_LABELS = ['zeroth', 'first', 'second', 'third', 'fourth', 'fifth']
ARRAY_TYPE = Union[Tensor, np.ndarray, List[Tensor], List[np.ndarray]]


def scatter(y: ARRAY_TYPE, x: Union[ARRAY_TYPE, str] = 'index', **kwargs) -> Optional[Figure]:
    fig, return_fig = create_generic_plot_fig(plot_fn=px.scatter, y=y, x=x, **kwargs)
    return fig if return_fig else fig.show()

def line(y: ARRAY_TYPE, x: Union[ARRAY_TYPE, str] = 'index', **kwargs) -> Optional[Figure]:
    fig, return_fig = create_generic_plot_fig(plot_fn=px.line, y=y, x=x, **kwargs)
    return fig if return_fig else fig.show()

def bar(y: ARRAY_TYPE, x: Union[ARRAY_TYPE, str] = 'index', **kwargs) -> Optional[Figure]:
    fig, return_fig = create_generic_plot_fig(plot_fn=px.bar, y=y, x=x, **kwargs)
    return fig if return_fig else fig.show()

def violin(y: ARRAY_TYPE, x: Optional[Union[ARRAY_TYPE, str]] = None, **kwargs) -> Optional[Figure]:
    fig, return_fig = create_generic_plot_fig(plot_fn=px.violin, y=y, x=x, **kwargs)
    return fig if return_fig else fig.show()

def box(y: ARRAY_TYPE, x: Optional[Union[ARRAY_TYPE, str]] = None, **kwargs) -> Optional[Figure]:
    fig, return_fig = create_generic_plot_fig(plot_fn=px.box, y=y, x=x, **kwargs)
    return fig if return_fig else fig.show()

def histogram(x: ARRAY_TYPE, **kwargs) -> Optional[Figure]:
    fig, return_fig = create_generic_plot_fig(plot_fn=px.histogram, x=x, **kwargs)
    return fig if return_fig else fig.show()



def create_generic_plot_fig(plot_fn: Callable[..., Figure], **kwargs) -> Tuple[Figure, bool]:
    plot_args = PlotArgs(plot_fn, **kwargs)
    
    df, broadcasted_shape = broadcast_to_df(plot_args.array_args) # , plot_args.custom_args.array_value_names
    df = add_labels_to_df(df, broadcasted_shape, plot_args.custom_args.dim_labels, plot_args.custom_args.value_names)
    plot_args.apply_args_from_df(df)

    fig = plot_fn(df, **plot_args.data_args)
    fig = fig.update_layout(**plot_args.layout_args)
    return fig, plot_args.custom_args.return_fig

def broadcast_to_df(array_args: Dict[str, ARRAY_TYPE], ) -> Tuple[pd.DataFrame, Tuple[int, ...]]:
    array_arg_names, arrays = zip(*array_args.items())
    broadcasted_arrays = np.broadcast_arrays(*[to_numpy(array) for array in arrays])
    df = pd.DataFrame(data={name: array.flatten() for name, array in zip(array_arg_names, broadcasted_arrays)})
    broadcasted_shape = broadcasted_arrays[0].shape
    return df, broadcasted_shape

def add_labels_to_df(df: pd.DataFrame,
                            broadcasted_array_shape: Tuple[int, ...],
                            dim_labels: List[str],
                            value_names: Optional[Dict[str, Any]], 
                            ) -> pd.DataFrame:
    labels = [[i for i in range(broadcasted_array_shape[dim])] for dim in range(len(broadcasted_array_shape))]
    dim_labels = dim_labels[:len(broadcasted_array_shape)]
    labels_cartesian_product = list(zip(*product(*labels)))

    df_labels = pd.DataFrame(data=labels_cartesian_product, index=dim_labels).T
    df = pd.concat([df_labels, df], axis=1)
    df = df.reset_index()
    df = apply_value_names_to_df(df, value_names)
    return df 

def apply_value_names_to_df(df: pd.DataFrame, value_names: Optional[Dict[str, Any]]) -> pd.DataFrame:
    for column_name, value_map in value_names.items():
        assert column_name in df.columns, f'Column name {column_name} from value_names arg not found in dataframe.'
        if isinstance(value_map, (list, dict)):
            try:
                df[column_name] = df[column_name].map(lambda x: value_map[x])
            except:
                raise ValueError(f'Value map for column {column_name} is not valid.') 
        else: 
            df[column_name] = pd.Categorical(df[column_name].map(value_map))
    return df


def to_numpy(array: ARRAY_TYPE) -> np.ndarray:
    if isinstance(array, list):
        return transform_array_list_to_numpy(array)
    else:
        return transformer_lens_to_numpy(array)

def transform_array_list_to_numpy(array_list: List[Tensor]) -> List[np.ndarray]:
    numpy_array_list = [transformer_lens_to_numpy(array) for array in array_list]
    first_array_shape = numpy_array_list[0].shape
    if all(array.shape == first_array_shape for array in numpy_array_list):
        return np.stack(numpy_array_list)
    
    elif all(len(array.shape) == len(first_array_shape) for array in numpy_array_list):
        print('Warning: A list of arrays was passed with arrays of different shapes. Arrays will be padded with nan to match the largest shape.')
        max_shape = [max([arr.shape[dim] for arr in numpy_array_list]) 
                     for dim in range(len(first_array_shape))]
        padded_array_list = [np.pad(arr, [(0, max_shape[dim] - arr.shape[dim]) for dim in range(len(arr.shape))],
                                    constant_values=np.nan) for arr in numpy_array_list]
        return np.stack(padded_array_list)
    else:
        raise ValueError('Arrays in list have different shapes and different number of dimensions.')
    
def is_array_type(arg_value: Any) -> bool: 
    is_array = isinstance(arg_value, (Tensor, np.ndarray))
    is_list_of_arrays = isinstance(arg_value, list) and isinstance(arg_value[0], (Tensor, np.ndarray))
    return is_array or is_list_of_arrays

class PlotArgs:
    def __init__(self, plot_fn: Callable[..., Figure], **kwargs):
        self.plot_fn = plot_fn
        self.plot_name = plot_fn.__name__
        self.data_args_set = set(inspect.signature(self.plot_fn).parameters.keys())
    
        self.array_args = self.pop_array_args(kwargs)
        self.custom_args = self.pop_custom_args(kwargs)
        self.data_args = self.pop_data_args(kwargs)
        self.layout_args = kwargs
    
        self.apply_default_args(self.plot_name)

    def pop_array_args(self, kwargs: Dict[str, Any]):
        return self.pop_args_on_condition(kwargs, lambda arg_name, arg_value: is_array_type(arg_value))
    
    def pop_custom_args(self, kwargs: Dict[str, Any]):
        cumstom_args_dict = self.pop_args_on_condition(kwargs, lambda arg_name, arg_value: arg_name 
                                                       in CustomArgs.__dataclass_fields__.keys())
        return CustomArgs(**cumstom_args_dict)
    
    def pop_data_args(self, kwargs: Dict[str, Any]):
        return self.pop_args_on_condition(kwargs, lambda arg_name, arg_value: arg_name in self.data_args_set)
    
    def pop_args_on_condition(self, kwargs: Dict[str, Any], condition: Callable[[str, Any], bool]):
        args = {arg_name: arg_value for arg_name, arg_value 
                in kwargs.items() if condition(arg_name, arg_value)}
        for arg_name in args.keys():
            kwargs.pop(arg_name)
        return args
    
    def apply_args_from_df(self, df: pd.DataFrame):
        self.set_data_arg_if_unspecified('hover_data', df.columns)

        self.overwrite_plotting_variables_with_column_names(df)
        self.order_plot_of_small_integer_columns(df)

    def order_plot_of_small_integer_columns(self, df):
        int_columns_order = {}
        for column_name in df.columns:
            if df[column_name].dtype == 'int64':
                unique_values = df[column_name].unique()
                if len(unique_values) < 10:
                    df[column_name] = pd.Categorical(df[column_name])
                    int_columns_order[column_name] = np.sort(df[column_name].unique())
            
        self.data_args['category_orders'].update(int_columns_order)

    def overwrite_plotting_variables_with_column_names(self, df):
        for column_name in df.columns:
            if column_name in self.data_args_set:
                self.data_args[column_name] = column_name
    
    def apply_default_args(self, plot_name: str):
        self.apply_shared_default_args()
        getattr(self, f'apply_default_{plot_name}_args')()

    def apply_shared_default_args(self):
        self.set_data_arg_if_unspecified('category_orders', {})

    def apply_default_scatter_args(self):
        self.set_data_arg_if_unspecified('symbol_sequence', ['circle', 'star', 'cross', 'square', 'diamond', 'x', 'triangle-up'])

    def apply_default_line_args(self):
        pass

    def apply_default_bar_args(self):
        pass
    
    def apply_default_violin_args(self):
        self.set_data_arg_if_unspecified('x', self.custom_args.dim_labels[0])

    def apply_default_box_args(self):
        self.set_data_arg_if_unspecified('x', self.custom_args.dim_labels[0])

    def apply_default_histogram_args(self):
        pass

    def set_data_arg_if_unspecified(self, arg_name: str, value: Any):
        if arg_name not in self.data_args or self.data_args[arg_name] is None:
            self.data_args[arg_name] = value


@dataclass
class CustomArgs:
    dim_labels: List[str] = field(default_factory=lambda: DEFAULT_DIM_LABELS)
    value_names: Dict[str, Any] = field(default_factory=dict)
    return_fig: bool = False
