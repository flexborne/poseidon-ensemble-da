"""Excel I/O contract tests (multi-sheet accumulation, named sheet)."""
from __future__ import annotations

import numpy as np
import pytest

from poseidon_da.paths import box_surfareas_with_centers_path
from poseidon_da.poseidon_io import load_numeric_excel, xlswrite


def test_xlswrite_preserves_multiple_sheets(tmp_path):
    from openpyxl import load_workbook
    path = tmp_path / 'multi.xlsx'
    xlswrite(str(path), np.array([[1.0], [2.0]]), 'a')
    xlswrite(str(path), np.array([[3.0], [4.0]]), 'b')
    xlswrite(str(path), np.array([[5.0]]), 'c')
    wb = load_workbook(path)
    assert {'a', 'b', 'c'}.issubset(set(wb.sheetnames))


def test_load_numeric_excel_named_sheet_and_missing():
    path = str(box_surfareas_with_centers_path())
    arr = load_numeric_excel(path, sheet_name='Sheet1')
    assert arr.ndim == 2 and arr.shape[0] > 0
    with pytest.raises(KeyError):
        load_numeric_excel(path, sheet_name='does-not-exist')
