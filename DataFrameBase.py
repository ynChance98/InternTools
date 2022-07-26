#!/usr/bin/env python
# coding: utf-8

# In[7]:


import logging
import pandas as pd
from typing import List


class DataFrameBase(object):
    """
    数据框基本操作封装，子类必须指定 _head _rename_dict 和 _var_type_dict 用来标准化列名和转换格式
    """
    _head = ()
    _rename_dict = {}
    _var_type_dict = {int: [], float: [], str: []}
    _index_name = None
    logger = logging.getLogger(__name__)

    def __init__(self, df):
        """
        :param df: 混合数据框
        """
        var_int_set = set(self._var_type_dict[int])
        var_float_set = set(self._var_type_dict[float])
        var_str_set = set(self._var_type_dict[str])
        assert (not var_int_set & var_str_set), f"{var_int_set} & {var_str_set} have duplicate value"
        assert (not var_int_set & var_float_set), f"{var_int_set} & {var_float_set} have duplicate value"
        assert (not var_float_set & var_str_set), f"{var_float_set} & {var_str_set} have duplicate value"

        if set(self._rename_dict.values()) <= set(self._head):
            self._df = self._format_head(df)
        else:
            raise KeyError(f"format_head: target df columns {self._head} "
                           f"must in {list(self._rename_dict.values())}")

        if set(self._head) == set(tuple(var_int_set) + tuple(var_float_set) + tuple(var_str_set)):
            self._df = self._transform_type(self._df, self._df.columns)
        else:
            print(self._head)
            print(tuple(var_int_set) + tuple(var_float_set) + tuple(var_str_set))
            raise KeyError("_var_type_dict must initialized by head.")

        if self._index_name is not None:
            self._df.set_index(self._index_name, inplace=True, drop=True)

    def _format_head(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._rename_dict is None:
            return df
        else:
            temp_rename_dict = {}
            for r in df.columns:
                if r in self._rename_dict:
                    temp_rename_dict[r] = self._rename_dict[r]
                else:
                    temp_rename_dict[r] = r
            assert set(temp_rename_dict.values()) <= set(self._head),                 f"{set(temp_rename_dict.values())} must <= {set(self._head)}"
            return df.rename(columns=temp_rename_dict)

    def _transform_type(self, df: pd.DataFrame, col: List[str]):
        if set(col) <= set(df.columns):
            df.loc[:, set(self._var_type_dict[int]) & set(col)].astype(int)
            df.loc[:, set(self._var_type_dict[float]) & set(col)].astype(float)
            df.loc[:, set(self._var_type_dict[str]) & set(col)].astype(str)
            return df
        else:
            raise KeyError(f"{col} must all in {df.columes}")

    def init_col(self, temp_df: pd.DataFrame):
        """
        添加当前df中不存在的列到df中，要求这一列必须在head中有定义并且index必须和当前df完全一致
        :param temp_df:
        :return:
        """
        temp_df = self._format_head(temp_df)
        if set(temp_df.columns) <= set(self._head):
            if any(map(lambda x: x in self._df.columns, temp_df.columns)):
                raise KeyError(f"init_col: {temp_df.columns} must out of {self._df.columns}")
        else:
            raise KeyError(f"init_col: {temp_df.columns} must all in {self._head}")

        if set(self._df.index) == set(temp_df.index):
            temp_df = self._transform_type(temp_df, temp_df.columns)
            self._df = pd.merge(self._df, temp_df, left_index=True, right_index=True)
        else:
            raise KeyError(f'temp_df index {set(temp_df.index)} must same with {set(self._df.index)}')

    def update_df(self, temp_df: pd.DataFrame, const_len_flag=True):
        """
        按照索引更新列数据 要求传入的数据索引必须和原数据框完全一致
        :param const_len_flag:
        :param temp_df: 新数据
        :return: 无
        """
        if self._index_name is not None and self._index_name in self._df.columns and self._df.empty is False:
            self._df.set_index(self._index_name)
        if const_len_flag:
            if set(self._df.index) == set(temp_df.index):
                temp_df = self._format_head(temp_df)
                temp_df = self._transform_type(temp_df, temp_df.columns)
                if set(temp_df.columns) <= set(self._df.columns):
                    self._df = self._df.drop(columns=temp_df.columns, axis=1).pd.merge(temp_df, on=self._index_name)
                else:
                    raise KeyError(f'My df columns {list(temp_df.columns)} must include all columns {self._df.columns}')
            else:
                raise KeyError(f'temp_df index {set(temp_df.index)} must same with {set(self._df.index)}')
        else:
            temp_df = self._format_head(temp_df)
            temp_df = self._transform_type(temp_df, temp_df.columns)
            if set(temp_df.columns) <= set(self._df.columns):
                self._df = self._df.drop(columns=temp_df.columns, axis=1).merge(
                    temp_df, on=self._index_name, how='outer')
                if self._index_name in self._df.columns:
                    self._df.set_index(self._index_name, inplace=True, drop=True)
            else:
                raise KeyError(f'My df columns {list(temp_df.columns)} must include all columns {self._df.columns}')

    def sorted(self, by: [list, str]):
        assert isinstance(by, (list, str))
        assert by in self._df.columns
        self._df.sort_values(by=by, inplace=True)

    def group(self, by: List[str]) -> dict:
        """
        根据by指定的顺序对数据框进行分组，并返回分组标准的索引升序排列的结果
        :param by:
        :return:
        """
        re = {}

        if set(by) <= set(self._df.columns):
            g = self._df.groupby(by=by)
            for ge in list(g.groups):
                re[ge] = list(g.get_group(ge).index)
            return re
        else:
            raise KeyError(f'My df columns {list(self._df.columns)} must include all columns {by}')

    @property
    def data(self):
        return self._df.copy()

    @property
    def head(self):
        return self._head

    @property
    def columns(self):
        return self._df.columns


# In[ ]:




